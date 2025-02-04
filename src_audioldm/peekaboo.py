import os
import sys

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src_audioldm')
sys.path.extend([proj_dir, src_dir])

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import rp
import src_audioldm.audioldm as ldm
from src_audioldm.utilities.data.dataset_pkb import AudioDataProcessor, spectral_normalize_torch
from src_audioldm.learnable_textures import (
    LearnableImageFourier,
    LearnableImageRasterSigmoided,
)

ldm = ldm.AudioLDM('cuda')
device = ldm.device

def make_learnable_image(height, width, num_channels, representation='fourier'):
    image_types = {
        'fourier': LearnableImageFourier(height, width, num_channels),
        'raster': LearnableImageRasterSigmoided(height, width, num_channels)}
    return image_types.get(representation, ValueError(f'Invalid method: {representation}'))

def blend_torch_images(foreground, background, alpha):
    'When alpha is scaler'
    return foreground * alpha + background * (1-alpha)

def masking_torch_image(foreground, alpha):
    assert foreground.shape == alpha.shape, 'foreground shape != alpha shape'
    assert alpha.min().item() >= 0 and alpha.max().item() <= 1, f'alpha range error {alpha.min().item()}, {alpha.max().item()}'
    min_val = foreground.min()
    blended = (foreground - min_val) * alpha + min_val
    return blended

class AudioPeekabooSeparator(nn.Module):
    def __init__(self,
                 dataset: dict,
                 processor=None,
                 min_step = None,
                 max_step = None,
                 representation: str = 'fourier bilateral',
                 device = 'cuda'):
        
        super().__init__()
        self.dataset = dataset
        self.height = dataset['stft'].shape[1]  # 513
        self.width = dataset['stft'].shape[2]   # 1024
        self.text = dataset['text'][0]
        self.processor = processor
        assert self.height == 513 and self.width == 1024, 'stft shape must be [B, 513, 1024]'
        
        self.device = device
        self.min_step = min_step
        self.max_step = max_step
        self.num_labels = 1
        self.representation = representation
        
        self.foreground = dataset['stft'].to(device)       
        self.background = torch.zeros_like(self.foreground)  # background is solid color now
        self.alphas = make_learnable_image(self.height, self.width, self.num_labels, representation)
        
    def forward(self, alphas=None, return_alphas=False):
        old_min_step, old_max_step = ldm.min_step, ldm.max_step
        if (self.min_step is not None) and (self.max_step is not None):
            ldm.min_step, ldm.max_step = self.min_step, self.max_step

        alphas = alphas if alphas is not None else self.alphas()  # alphas: [1, 513, 1024]
        assert alphas.shape==(self.num_labels, self.height, self.width)
        assert alphas.min()>=0 and alphas.max()<=1
        
        masked_stft = masking_torch_image(self.foreground, alphas)
        self.data['stft'] = masked_stft.float()

        mel_basis = self.processor.mel_basis[f"{self.dataset.mel_fmax}_{self.device}"].to(self.device)
        mel = spectral_normalize_torch(torch.matmul(mel_basis, masked_stft))
        
        masked_log_mel_spec = self.processor.pad_spec(mel[0].T).unsqueeze(0).float()
        original_shape = self.dataset['log_mel_spec'].shape
        assert self.dataset['log_mel_spec'].shape == masked_log_mel_spec.shape, f'{original_shape} != {masked_log_mel_spec.shape}'
        self.dataset['log_mel_spec'] = masked_log_mel_spec

        assert not torch.isnan(alphas).any() or not torch.isinf(alphas).any(), "alpha contains NaN or Inf values"  # NaN이나 Inf 값 체크

        return (self.dataset, alphas) if return_alphas else self.dataset


class Maskgenerator(nn.Module):
   def __init__(self):
       super().__init__()
       self.prm = nn.Parameter(0.1 * torch.randn(1))
       
   def forward(self) -> torch.Tensor:
       return torch.sigmoid(self.prm)

class TestSeparator(nn.Module):
    def __init__(self,
                 dataset: dict,
                 datase2: dict,
                 processor=None,
                 device = 'cuda'):
        
        super().__init__()
        self.dataset = dataset
        self.datase2 = datase2
        self.text = dataset['text']
        self.processor = processor
        self.device = device
        
        self.foreground = dataset['log_mel_spec'].to(device)
        self.background = datase2['log_mel_spec'].to(device)
        self.alphas = Maskgenerator().to(device)
        
    def forward(self, alphas=None, return_alphas=False):
        alphas = alphas if alphas is not None else self.alphas()  # alphas: [1, 513, 1024]
        # assert alphas.min()>=0 and alphas.max()<=1
        fmin, fmax = self.foreground.min(), self.foreground.max()
        bmin, bmax = self.background.min(), self.background.max()
        # assert fmin == bmin, f'foreground min != background min {fmin} != {bmin}'
        # assert self.foreground.min()-fmin == self.background.min() - bmin == 0 == (self.foreground - fmin).min()==(self.background - bmin).min(),\
        'foreground - min != background - min'
        print(alphas)

        masked_mel = ((self.foreground - fmin) * alphas) + ((self.background - fmin) * (1 - alphas)) + fmin
        mmin, mmax = masked_mel.min(), masked_mel.max()
        # assert mmin == fmin, f'masked min != foreground min {mmin} != {fmin}'
        assert masked_mel.shape == self.foreground.shape, f'{masked_mel.shape} != {self.foreground.shape}'
        # if fmax > bmax:
        #     assert mmax <= fmax, f'masked max > foreground max {mmax} > {fmax}'
        # else:
        #     assert mmax <= bmax, f'masked max > background max {mmax} > {bmax}'        
        assert not torch.isnan(alphas).any() or not torch.isinf(alphas).any(), "alpha contains NaN or Inf values"  # NaN이나 Inf 값 체크
        self.dataset['log_mel_spec'] = masked_mel
        return (self.dataset, alphas) if return_alphas else self.dataset


def run_peekaboo(target_text: str,
                 audio_file_path: str,
                 GRAVITY=1e-1/2,      # prompt에 따라 tuning이 제일 필요. (1e-2, 1e-1/2, 1e-1, 1.5*1e-1)
                 NUM_ITER=300,        # 이정도면 충분
                 LEARNING_RATE=1e-5,  # neural neural texture 아니면 키워도 됨.
                 BATCH_SIZE=1,        # 키우면 vram만 잡아먹음
                 GUIDANCE_SCALE=100,  # DreamFusion 참고하여 default값 설정
                 representation='fourier bilateral',
                 min_step=None,
                 max_step=None):

    audioprocessor = AudioDataProcessor(device=device)
    dataset = audioprocessor.preprocessing_data(audio_file_path)
    dataset2 = audioprocessor.preprocessing_data("./best_samples/Footsteps_on_a_wooden_floor.wav")

    '''
    {
    "text":         # list
    "fname":        # list
    "waveform":     # tensor, [B, 1, samples_num]
    "stft":         # tensor, [B, t-steps, f-bins]
    "log_mel_spec": # tensor, [B, t-steps, mel-bins]
    }
    '''
    assert len(dataset['text']) == 1, 'Only one audio file is allowed'
    # assert dataset['text'][0] == target_text, 'Text does not match'
    if dataset['text'][0] != target_text:
        dataset['text'][0] = target_text
        print("Warning!: Text has been changed to match the target_text")

    # pkboo = AudioPeekabooSeparator(
    #     dataset,
    #     processor=audioprocessor,
    #     representation=representation,
    #     min_step=min_step,
    #     max_step=max_step,
    #     device=device
    # ).to(device)

    pkboo = TestSeparator(
        dataset,
        dataset2,
        processor=audioprocessor,
        device=device
    ).to(device)

    optim = torch.optim.SGD(list(pkboo.parameters()), lr=LEARNING_RATE)

    def train_step():
        alphas = pkboo.alphas()
        composite_set = pkboo()
        
        dummy_for_plot = ldm.train_step(composite_set, guidance_scale=GUIDANCE_SCALE)
        
        loss = alphas.mean() * GRAVITY
        alphaloss = loss.item()
        alphaloss = alphas.mean().item()
        # loss2 = torch.abs(alphas[:, 1:, :] - alphas[:, :-1, :]).mean() + torch.abs(alphas[:, :, 1:] - alphas[:, :, :-1]).mean()
        # loss += loss2 * 5000
        # print(loss2.item())
        loss.backward(); optim.step(); optim.zero_grad()
        sdsloss, uncond, cond, eps_diff = dummy_for_plot
        return sdsloss, alphaloss, uncond, cond, eps_diff

    list_sds, list_alpha, list_uncond_eps, list_cond_eps, list_eps_differ = [], [], [], [], []
    list_dummy = (list_sds, list_alpha, list_uncond_eps, list_cond_eps, list_eps_differ)
    try:
        for iter_num in tqdm(range(NUM_ITER)):
            dummy_for_plot = train_step()
            for li, element in zip(list_dummy, dummy_for_plot):
                li.append(element)

    except KeyboardInterrupt:
        print("Interrupted early, returning current results...")
        pass
    
    alphas = pkboo.alphas()

    def save_melspec_as_img(mel_tensor, save_path):
        mel = mel_tensor.detach().cpu().numpy()
        if mel.shape[0] > mel.shape[1]:
            mel = mel.T  # (64, 1024)로 전치
        height, width = mel.shape
        aspect_ratio = width / height  # 1024/64 = 16
        fig_width = 20  # 기준 가로 길이
        fig_height = fig_width / aspect_ratio  # 20/16 = 1.25
        if mel.min() < 0:
            # min_, max_ = -11.5129, 3.4657
            min_, max_ = mel.min(), mel.max()
        else:
            min_, max_ = 0, 1
        plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(mel, aspect='auto', origin='lower', cmap='magma',
                vmin=min_, vmax=max_)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    results = {
        "alphas":rp.as_numpy_array(alphas),
        
        "representation":representation,
        "NUM_ITER":NUM_ITER,
        "GRAVITY":GRAVITY,
        "lr":LEARNING_RATE,
        "GUIDANCE_SCALE":GUIDANCE_SCALE,
        "BATCH_SIZE":BATCH_SIZE,

        "alpha": alphas.mean().item(),
        
        "target_text":pkboo.text[0],
        "device":device,
    }

    output_folder = rp.make_folder('peekaboo_results/%s'%target_text)
    output_folder += '/%03i'%len(rp.get_subfolders(output_folder))
    save_peekaboo_results(results, output_folder, list_dummy)
    print(f"Saved results at {output_folder}")

    mel_cpu = dataset['log_mel_spec'][0, ...].detach().cpu()
    save_melspec_as_img(mel_cpu, os.path.join(output_folder, "orign_mel.png"))

    mel_cpu = pkboo(alphas=torch.Tensor([0.5]).to(device))['log_mel_spec'][0, ...].detach().cpu()
    save_melspec_as_img(mel_cpu, os.path.join(output_folder, "mixed_mel.png"))

    mel_cpu = pkboo()['log_mel_spec'][0, ...].detach().cpu()
    save_melspec_as_img(mel_cpu, os.path.join(output_folder, "seped_mel.png"))

def save_peekaboo_results(results, new_folder_path, list_dummy):
    import json
    assert not rp.folder_exists(new_folder_path), f'Please use a different name, not {new_folder_path}'
    rp.make_folder(new_folder_path)
    with rp.SetCurrentDirectoryTemporarily(new_folder_path):
        print(f"Saving PeekabooResults to {new_folder_path}")
        params = {}
        for key, value in results.items():
            if rp.is_image(value):  # Save a single image
                rp.save_image(value, f'{key}.png')
            elif isinstance(value, np.ndarray) and rp.is_image(value[0]):  # Save a folder of images
                rp.make_directory(key)
                with rp.SetCurrentDirectoryTemporarily(key):
                    for i in range(len(value)):
                        rp.save_image(value[i], f'{i}.png')
            elif isinstance(value, np.ndarray):  # Save a generic numpy array
                np.save(f'{key}.npy', value) 
            else:
                try:
                    json.dumps({key: value})
                    params[key] = value  #Assume value is json-parseable
                except Exception:
                    params[key] = str(value)
        rp.save_json(params, 'params.json', pretty=True)
        print(f"Done saving PeekabooResults to {new_folder_path}!")
    
    # Loss plot 저장
    sds, alpha, uncond, cond, eps = list_dummy
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1); plt.plot(sds, label='SDS Loss')
    plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.legend()
    plt.subplot(2, 1, 2); plt.plot(alpha, label='Alpha Loss')
    plt.xlabel('Iteration'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(); plt.savefig(f'{new_folder_path}/loss_plot.png'); plt.close()

    plt.figure(figsize=(25, 10))
    plt.subplot(3, 1, 1); plt.plot(uncond, label='uncond')
    plt.xlabel('Iteration'); plt.ylabel('abs mean'); plt.legend()
    plt.subplot(3, 1, 2); plt.plot(cond, label='cond')
    plt.xlabel('Iteration'); plt.ylabel('abs mean'); plt.legend()
    plt.subplot(3, 1, 3); plt.plot(eps, label='difference bet eps')
    plt.xlabel('Iteration'); plt.ylabel('abs mean'); plt.legend()
    plt.tight_layout(); plt.savefig(f'{new_folder_path}/eps_plot.png'); plt.close()

if __name__ == "__main__":
    
    # # bilateral fourier 용도.
    # prms = {
    #     'G': 3000,
    #     'iter': 300,
    #     'lr': 1e-5,
    #     'B': 1,
    #     'guidance': 100,
    #     'representation': 'fourier bilateral',
    # }

    # raster 용도.
    prms = {
        'G': 0, # 3000,
        'iter': 100,
        'lr': 0.0001,
        'B': 1,
        'guidance': 100,
        'representation': 'raster',
    }

    run_peekaboo(
        target_text='Footsteps on a wooden floor', # 'A cat meowing',
        audio_file_path="./best_samples/A_cat_meowing.wav",
        GRAVITY=prms['G'],
        NUM_ITER=prms['iter'],
        LEARNING_RATE=prms['lr'],
        BATCH_SIZE=prms['B'],
        GUIDANCE_SCALE=prms['guidance'],
        representation=prms['representation'],
        )
    
    run_peekaboo(
        target_text='A cat meowing', # 'A cat meowing',
        audio_file_path="./best_samples/A_cat_meowing.wav",
        GRAVITY=prms['G'],
        NUM_ITER=prms['iter'],
        LEARNING_RATE=prms['lr'],
        BATCH_SIZE=prms['B'],
        GUIDANCE_SCALE=prms['guidance'],
        representation=prms['representation'],
        )
