import os
import sys
import re
from typing import Dict, List
import traceback

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src')
sys.path.extend([proj_dir, src_dir])

import matplotlib.pyplot as plt

import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pathlib
import librosa
import yaml
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
# import lightning.pytorch as pl
# from models.clap_encoder import CLAP_Encoder

import soundfile as sf
from src.audioldm import AudioLDM as ldm
from src.utilities.data.dataprocessor import AudioDataProcessor as prcssr

import torchaudio

def load_audio_torch(source_path, sampling_rate, mono=True):
    waveform, sr = torchaudio.load(source_path, normalize=True)  # librosa처럼 float32 [-1, 1]로 로드
    waveform = waveform.mean(dim=0) if (waveform.shape[0] > 1) and mono else waveform  # mono 변환
    waveform = torchaudio.functional.resample(waveform, sr, sampling_rate) if sr != sampling_rate else waveform
    return waveform.numpy().squeeze(0), sampling_rate

def calculate_sdr(ref: np.ndarray, est: np.ndarray, eps=1e-10) -> float:
    r"""Calculate SDR between reference and estimation.
    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    reference = ref
    noise = est - reference
    numerator = np.clip(a=np.mean(reference ** 2), a_min=eps, a_max=None)
    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)
    sdr = 10. * np.log10(numerator / denominator)
    return sdr

def calculate_sisdr(ref, est):
    r"""Calculate SDR between reference and estimation.
    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    eps = np.finfo(ref.dtype).eps
    reference = ref.copy()
    estimate = est.copy()
    reference = reference.reshape(reference.size, 1)
    estimate = estimate.reshape(estimate.size, 1)
    Rss = np.dot(reference.T, reference)
    # get the scaling factor for clean sources
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)
    e_true = a * reference
    e_res = estimate - e_true
    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()
    sisdr = 10 * np.log10((eps+ Sss)/(eps + Snn))
    return sisdr 

def get_mean_sdr_from_dict(sdris_dict):
    mean_sdr = np.nanmean(list(sdris_dict.values()))
    return mean_sdr

def parse_yaml(config_yaml: str) -> Dict:
    r"""Parse yaml file.
    Args:
        config_yaml (str): config yaml path
    Returns:
        yaml_dict (Dict): parsed yaml file
    """
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


class Mask(nn.Module):
    def __init__(self, device, channel=1, height=1024, width=513):
        super().__init__()
        # self.weight = nn.Parameter(torch.randn((channel, height, width)))
        self.weight = nn.Parameter(torch.full((channel, height, width), 3.0))
        self.to(device)
        self.device = device
    def forward(self):
        return torch.sigmoid(self.weight)


class AudioCapsEvaluator:
    def __init__(self, query='caption', sampling_rate=32000) -> None:
        r"""AudioCaps evaluator.
        Args:
            query (str): type of query, 'caption' or 'labels'
        Returns:
            None
        """
        self.query = query
        self.sampling_rate = sampling_rate
        with open(f'evaluation/metadata/audiocaps_eval.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            eval_list = [row for row in csv_reader][1:]
        self.eval_list = eval_list
        self.audio_dir = f'evaluation/data/audiocaps'

    def __call__(self, pl_model, config) -> Dict:
        r"""Evalute."""
        print(f'Evaluation on AudioCaps with [{self.query}] queries.')
        
        # pl_model.eval()

        processor, aldm = pl_model
        device = aldm.device

        sisdrs_list = []
        sdris_list = []
        
        whentobreak = config['break']

        loss_values = []  # 손실값 저장 리스트
        import matplotlib.pyplot as plt


        '''        
        eval_data = self.eval_list[0]
        idx, caption, labels, _, _ = eval_data

        source_path = os.path.join(self.audio_dir, f'segment-{idx}.wav')

        source_path = '/home/sba2024/MIIL_ZSS/audio_peekaboo/a_cat_n_stepping_wood.wav'
        mel_src, _, stft_complex_src, wav_src, _ = processor.read_audio_file(source_path)
        text = [caption]
        wav_sep = processor.inverse_mel_with_phase(
            mel_src,
            stft_complex_src[:,:,:1024],
        )

        wav_src = wav_src.squeeze(0).data.cpu().numpy()
        wav_sep = wav_sep.squeeze(0).data.cpu().numpy()
        
        wav_src = wav_src[:len(wav_sep)]

        sisdr = calculate_sisdr(wav_src, wav_sep)
        print(sisdr)
        sf.write('./istft.wav',wav_sep,16000)


        mel_src, _, stft_complex_src, wav_src, _ = processor.read_audio_file(source_path)
        text = [caption]
        wav_sep = aldm.mel_to_waveform(mel_src)
        
        al_src = wav_src.squeeze(0).data.cpu().detach().numpy()
        al_sep = wav_sep.squeeze(0).data.cpu().detach().numpy()
        
        import numpy as np
        import scipy.signal

        def time_align_and_trim(signal1, signal2):
            corr = scipy.signal.correlate(signal1, signal2, mode="full")
            shift = np.argmax(corr) - (len(signal2) - 1)
            aligned_signal2 = np.roll(signal2, shift)

            if shift > 0:
                aligned_signal1 = signal1[shift:]
                aligned_signal2 = aligned_signal2[:len(aligned_signal1)]
            else:
                aligned_signal2 = aligned_signal2[-shift:]
                aligned_signal1 = signal1[:len(aligned_signal2)]

            return aligned_signal1, aligned_signal2


        al_src, al_sep = time_align_and_trim(al_src, al_sep)

        

        sisdr = calculate_sisdr(al_src, al_sep)
        print(sisdr)
        sf.write('./vocoder.wav',al_sep,16000)


        raise ValueError
        '''

################################################
        for i in range(10):
            mask = Mask(device, 1, 513, 1024)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(mask.parameters(), lr=0.01)

            src_path = './samples/best_samples/A_cat_meowing.wav'
            mixed_path = './a_cat_n_stepping_wood.wav'

            _, _, _, wav_src, _ = processor.read_audio_file(src_path, pad_stft=False)
            mel_mix, stft_mix, stft_complex_mix, wav_mix, _ = processor.read_audio_file(mixed_path)
            

            wav_src = wav_src.squeeze(0).data.cpu().numpy()
            wav_mix = wav_mix.squeeze(0).data.cpu().numpy()

            sdr_no_sep = calculate_sdr(ref=wav_src, est=wav_mix)
            sdr = calculate_sdr(ref=wav_src, est=wav_mix)
            sdri = sdr - sdr_no_sep
            sisdr = calculate_sisdr(ref=wav_src, est=wav_mix)
            print(sisdr, sdri)


            for param in aldm.parameters():
                param.requires_grad = False  # 모든 파라미터를 학습 불가능하게 고정

            mel_tar_samples = []
            for ii in tqdm(range(3)):
                mel_sample = aldm.edit_audio_with_ddim(
                                    mel=mel_mix,
                                    text=str(f'A cat meowing'),
                                    duration=10.24,
                                    batch_size=1,
                                    transfer_strength=0.5,
                                    guidance_scale=2.5,
                                    ddim_steps=200,
                                    clipping = False,
                                    return_type="mel",
                                )
                mel_tar_samples.append(mel_sample)

                wav_sample = processor.inverse_mel_with_phase(
                            mel_sample,
                            stft_complex_mix[:,:,:1024],
                        )

                pad_size = 432
                if wav_sample.shape[-1] > pad_size*2:
                    wav_sample = wav_sample[..., pad_size:-pad_size]  # shape [B, samples]
                else:
                    # 혹시 길이가 매우 짧다면 예외처리
                    wav_sample = wav_sample[..., 0:1]

                wav_sep = wav_sample.squeeze(0).data.cpu().numpy()

                sf.write(f'./edit{ii}.wav', wav_sep, 16000)






            # mel_tar = torch.stack(mel_tar_samples).mean(dim=0)  # 평균 계산
            mel_tar = torch.stack(mel_tar_samples)  # 평균 계산

        
            num_epochs = 3000
            for epoch in range(num_epochs):

                optimizer.zero_grad()  # 그래디언트 초기화

                st = stft_mix.squeeze(0).permute(0,2,1)  #ts[1,513,1024]
                masked_stft = (st - st.min()) * mask() + st.min()  #ts[1,513,1024]
                
                log_mel_spec, _, _ = processor.stft_to_mel(masked_stft, stft_complex_mix)  # ts[1, M:64, T:1024~] / ts[1, F:513, T:1024~] / ts[1, F:513, T:1024~]
                log_mel_spec, _ = processor.postprocess_spec(log_mel_spec)
                mel_mix = log_mel_spec[None, None, ...]  # [1,1,1024,512]
                mel_mix_expanded = mel_mix.repeat(30, 1, 1, 1)  # [1,1,1024,512]

                loss = criterion(mel_tar, mel_mix_expanded)  # 손실 계산

                loss.backward()  # 역전파
                optimizer.step()  # 가중치 업데이트

                loss_values.append(loss.item())  # 손실값 저장

                if epoch % 100 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")


            plt.plot(loss_values)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Trend')
            plt.savefig('./plot.png')

            mask = mask()
            threshold = 0.9
            # mask = torch.where(mask > threshold, torch.tensor(1.0, device=mask.device), mask)
            masked_stft_mag = st * mask

            eps=1e-5
            phase = stft_complex_mix / (stft_complex_mix.abs() + eps)
            masked_stft_complex = masked_stft_mag * phase[:,:,:1024]

            estimated_wav = torch.istft(
                masked_stft_complex.to(device),
                n_fft=1024,
                hop_length=160,
                win_length=1024,
                window=torch.hann_window(1024).to(device),
                normalized=False,
                onesided=True
            )  # shape [B, samples + 2*pad_size]

            # 5) pad_size 부분 제거
            #    forward에서 reflect pad를 (pad_size, pad_size)만큼 했었으므로,
            #    최종 waveform에서 앞뒤로 pad_size samples씩 잘라낸다.
            pad_size = 432
            if estimated_wav.shape[-1] > pad_size*2:
                estimated_wav = estimated_wav[..., pad_size:-pad_size]  # shape [B, samples]
            else:
                # 혹시 길이가 매우 짧다면 예외처리
                estimated_wav = estimated_wav[..., 0:1]

            wav_sep = estimated_wav.squeeze(0).data.cpu().numpy()

            sf.write('./mse.wav', wav_sep, 16000)

            # wav_src = wav_src.squeeze(0).data.cpu().numpy()
            # wav_mix = wav_mix.squeeze(0).data.cpu().numpy()

            assert len(wav_sep) <= len(wav_mix), len(wav_sep)
            wav_src = wav_src[:len(wav_sep)]

            wav_mix = wav_mix[:len(wav_sep)]

            sdr_no_sep = calculate_sdr(ref=wav_src, est=wav_mix)
            sdr = calculate_sdr(ref=wav_src, est=wav_sep)
            sdri = sdr - sdr_no_sep
            sisdr = calculate_sisdr(ref=wav_src, est=wav_sep)

            print(sisdr, sdri)
################################################
        raise ValueError
################################################
        mask = Mask(device, 1, 513, 1024)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mask.parameters(), lr=0.01)

        src_path = './samples/best_samples/A_cat_meowing.wav'
        mixed_path = './mse.wav'

        mel_src, stft_src, stft_complex_src, wav_src, _ = processor.read_audio_file(src_path, pad_stft=False)
        mel_mix, stft_mix, stft_complex_mix, wav_mix, _ = processor.read_audio_file(mixed_path)

        mel_tar = aldm.edit_audio_with_ddim(
                                mel=mel_mix,
                                text=str(f'A cat meowing'),
                                duration=10.24,
                                batch_size=1,
                                transfer_strength=0.5,
                                guidance_scale=2.5,
                                ddim_steps=50,
                                clipping = False,
                                return_type="mel",
                            )

        num_epochs = 300
        for epoch in range(num_epochs):
            optimizer.zero_grad()  # 그래디언트 초기화

            st = stft_mix.squeeze(0).permute(0,2,1)  #ts[1,513,1024]
            masked_stft = st * mask()  #ts[1,513,1024]
            
            log_mel_spec, _, _ = processor.stft_to_mel(masked_stft, stft_complex_mix)  # ts[1, M:64, T:1024~] / ts[1, F:513, T:1024~] / ts[1, F:513, T:1024~]
            log_mel_spec, _ = processor.postprocess_spec(log_mel_spec)
            mel_mix = log_mel_spec[None, None, ...]



            loss = criterion(mel_tar, mel_mix)  # 손실 계산

            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트

            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")


        mask = mask()
        threshold = 0.9
        # mask = torch.where(mask > threshold, torch.tensor(1.0, device=mask.device), mask)
        masked_stft_mag = st * mask

        eps=1e-5
        phase = stft_complex_mix / (stft_complex_mix.abs() + eps)
        masked_stft_complex = masked_stft_mag * phase[:,:,:1024]

        estimated_wav = torch.istft(

            masked_stft_complex.to(device),
            n_fft=1024,
            hop_length=160,
            win_length=1024,
            window=torch.hann_window(1024).to(device),
            normalized=False,
            onesided=True
        )  # shape [B, samples + 2*pad_size]

        # 5) pad_size 부분 제거
        #    forward에서 reflect pad를 (pad_size, pad_size)만큼 했었으므로,
        #    최종 waveform에서 앞뒤로 pad_size samples씩 잘라낸다.
        pad_size = 432
        if estimated_wav.shape[-1] > pad_size*2:
            estimated_wav = estimated_wav[..., pad_size:-pad_size]  # shape [B, samples]
        else:
            # 혹시 길이가 매우 짧다면 예외처리
            estimated_wav = estimated_wav[..., 0:1]

        wav_sep = estimated_wav.squeeze(0).data.cpu().numpy()

        sf.write('./mse2.wav', wav_sep, 16000)

        wav_src = wav_src.squeeze(0).data.cpu().numpy()
        wav_mix = wav_mix.squeeze(0).data.cpu().numpy()

        assert len(wav_sep) <= len(wav_mix), len(wav_sep)
        wav_src = wav_src[:len(wav_sep)]

        wav_mix = wav_mix[:len(wav_sep)]

        sdr_no_sep = calculate_sdr(ref=wav_src, est=wav_mix)
        sdr = calculate_sdr(ref=wav_src, est=wav_sep)
        sdri = sdr - sdr_no_sep
        sisdr = calculate_sisdr(ref=wav_src, est=wav_sep)

        print(sisdr, sdri)
################################################






        import matplotlib.pyplot as plt
        import scipy.io.wavfile as wav
        import librosa.display as ld  

        def plot_wav_mel(wav_paths, save_path="./test/waveform_mel2.png"):
            fig, axes = plt.subplots(2, len(wav_paths), figsize=(4 * len(wav_paths), 6))

            for i, wav_path in enumerate(wav_paths):
                sr, data = wav.read(wav_path)
                time = np.linspace(0, len(data) / sr, num=len(data))
                
                # Waveform
                axes[0, i].plot(time, data, lw=0.5)
                axes[0, i].set_title(f"Waveform {i+1}")
                axes[0, i].set_xlabel("Time (s)")
                axes[0, i].set_ylabel("Amplitude")

                # Mel Spectrogram
                y, sr = librosa.load(wav_path, sr=None)
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                ld.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1, i])
                axes[1, i].set_title(f"Mel Spectrogram {i+1}")

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        wavs = ['./samples/best_samples/A_cat_meowing.wav',
                './a_cat_n_stepping_wood.wav',
                './mse.wav',
                './mse2.wav',]

        plot_wav_mel(wavs)



        raise ValueError

        try:
            with torch.no_grad():
                for i, eval_data in tqdm(enumerate(self.eval_list), total=len(self.eval_list)):
                    if i == whentobreak:
                        break
                    idx, caption, labels, _, _ = eval_data

                    source_path = os.path.join(self.audio_dir, f'segment-{idx}.wav')
                    mixture_path = os.path.join(self.audio_dir, f'mixture-{idx}.wav')

                    mel_src, _, stft_complex_src, wav_src, _ = processor.read_audio_file(source_path)
                    mel_mix, _, stft_complex_mix, wav_mix, _ = processor.read_audio_file(mixture_path)

                    if self.query == 'caption':
                        text = [caption]
                    elif self.query == 'labels':
                        text = [labels]

                    ts = config['transfer_strength']
                    guid = config['guidance_scale']
                    totalstep = config['ddim_steps']
                    iteration = config['iteration']
                    do_clip = config['do_clip']
                    iSTFT = config['iSTFT']
                    duration = 10.24

                    if iSTFT:
                        current_mel = mel_mix
                        for iter in range(iteration):
                            current_mel = aldm.edit_audio_with_ddim(
                                mel=current_mel,
                                text=str(f'Nothing but {text[0]}'),
                                duration=duration,
                                batch_size=1,
                                transfer_strength=ts,
                                guidance_scale=guid,
                                ddim_steps=totalstep,
                                clipping = do_clip,
                                return_type="mel",
                            )

                        wav_sep = processor.inverse_mel_with_phase(
                            masked_mel_spec=current_mel,
                            stft_complex=stft_complex_mix[:,:,:1024],
                        )

                        wav_src = wav_src.squeeze(0).data.cpu().numpy()
                        wav_mix = wav_mix.squeeze(0).data.cpu().numpy()
                        wav_sep = wav_sep.squeeze(0).data.cpu().numpy()
                        
                        assert len(wav_sep) <= len(wav_mix), len(wav_sep)
                        wav_src = wav_src[:len(wav_sep)]
                        wav_mix = wav_mix[:len(wav_sep)]
                        assert wav_src.shape == wav_mix.shape == wav_sep.shape, f"{wav_src.shape}, {wav_mix.shape}, {wav_sep.shape}"

                    else:
                        current_mel = mel_mix
                        for iter in range(iteration):
                            last_waveform = aldm.edit_audio_with_ddim(
                                mel=current_mel,
                                text=str(f'Nothing but {text[0]}'),
                                duration=duration,
                                batch_size=1,
                                transfer_strength=ts,
                                guidance_scale=guid,
                                ddim_steps=totalstep,
                                clipping = do_clip,
                                return_type="np",
                            )
                            last_waveform = torch.FloatTensor(last_waveform)
                            mel, stft, stft_com = processor.wav_feature_extraction(last_waveform)
                            current_mel = mel
                        wav_sep = last_waveform

                    sdr_no_sep = calculate_sdr(ref=wav_src, est=wav_mix)
                    sdr = calculate_sdr(ref=wav_src, est=wav_sep)
                    sdri = sdr - sdr_no_sep
                    sisdr = calculate_sisdr(ref=wav_src, est=wav_sep)
                    
                    sisdrs_list.append(sisdr)
                    sdris_list.append(sdri)
                    
                    if True:
                        print(text[0])
                        t = text[0]
                        text = re.sub(r'[\/:*?"<>| ]', '_', t)

                        sf.write(f'./z_result/{i}_result_{t}.wav', wav_sep, 16000)
                        sf.write(f'./z_result/{i}_gt_{t}.wav', wav_src, 16000)
                        sf.write(f'./z_result/{i}_mixture_{t}.wav', wav_mix, 16000)


        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

        finally:
            mean_sisdr = np.mean(sisdrs_list)
            mean_sdri = np.mean(sdris_list)
            
            return mean_sisdr, mean_sdri

if __name__ == "__main__":
    def clean_wav_filenames(dir_path):
        if not os.path.exists(dir_path):
            return
        for filename in os.listdir(dir_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(dir_path, filename)
                os.remove(file_path)

    clean_wav_filenames("./z_result")

    eval = AudioCapsEvaluator(query='caption', sampling_rate=16000)
    
    aldm = ldm('cuda:0')
    device = aldm.device
    processor = prcssr(device=device)

    config = {
        'transfer_strength': 0.2,
        'ddim_steps': 200,
        'guidance_scale': 2.5,
        'iteration': 4,
        'do_clip': False,
        'iSTFT': True,
        'break': 30,
    }

    mean_sisdr, mean_sdri = eval((processor, aldm), config)
    
    print(" SI-SDR  |  SDRi ")
    print(f"{round(mean_sisdr, 2)}  |  {round(mean_sdri, 2)}")






    # import matplotlib.pyplot as plt
    # import scipy.io.wavfile as wav
    # import librosa.display as ld  

    # def plot_wav_mel(wav_paths, save_path="./test/waveform_mel.png"):
    #     fig, axes = plt.subplots(2, len(wav_paths), figsize=(4 * len(wav_paths), 6))

    #     for i, wav_path in enumerate(wav_paths):
    #         sr, data = wav.read(wav_path)
    #         time = np.linspace(0, len(data) / sr, num=len(data))
            
    #         # Waveform
    #         axes[0, i].plot(time, data, lw=0.5)
    #         axes[0, i].set_title(f"Waveform {i+1}")
    #         axes[0, i].set_xlabel("Time (s)")
    #         axes[0, i].set_ylabel("Amplitude")

    #         # Mel Spectrogram
    #         y, sr = librosa.load(wav_path, sr=None)
    #         mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    #         mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    #         ld.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", ax=axes[1, i])
    #         axes[1, i].set_title(f"Mel Spectrogram {i+1}")

    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()

    # wavs = ['./test/origin.wav',
    #         './test/time_align_shit.wav',
    #         './test/orig.wav',
    #         './test/noised.wav',]

    # plot_wav_mel(wavs)
