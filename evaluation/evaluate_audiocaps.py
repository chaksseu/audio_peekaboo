import os
import sys
import re
from typing import Dict, List
# import traceback

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src')
sys.path.extend([proj_dir, src_dir])

import csv
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import pathlib
import librosa
import yaml
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

        pcr, aldm = pl_model
        device = aldm.device

        sisdrs_list = []
        sdris_list = []
        sisdrs_list_with_mix = []
        sdris_list_with_mix = []

        # try:
        with torch.no_grad():
            for i, eval_data in tqdm(enumerate(self.eval_list)):

                idx, caption, labels, _, _ = eval_data

                source_path = os.path.join(self.audio_dir, f'segment-{idx}.wav')
                mixture_path = os.path.join(self.audio_dir, f'mixture-{idx}.wav')

                # source, fs = load_audio_torch(source_path, sampling_rate=self.sampling_rate, mono=True)  # np[N,]
                # mixture, fs = load_audio_torch(mixture_path, sampling_rate=self.sampling_rate, mono=True)

                _, _, _, source, rand_start = pcr.read_audio_file(source_path)  # np[1,N,]
                mixture, rand_start = pcr.read_wav_file(mixture_path)



                logm, stft, stft_c = pcr.wav_feature_extraction(source)
                print(logm.shape)
                print(stft_c.shape)

                source_ = pcr.inverse_mel_with_phase(logm, stft_c.squeeze(0))
                
                sdr = calculate_sisdr(source,source_)
                print(sdr)
                
                raise ValueError
                ########################
                current_wav = mixture
                for i in range(1):
                    mixture_mel, _ = pcr.wav_feature_extraction(current_wav, pad_stft=True)
                    edited_waveform = aldm.mel_to_waveform(mixture_mel)
                    current_wav = edited_waveform[None, ...]

                    sdr = calculate_sisdr(mixture,current_wav)
                    print(sdr)
                    sdr = calculate_sisdr(mixture,mixture)
                    print(sdr)
                

                curr = current_wav.squeeze(0)
                mix = mixture.squeeze(0)


                import matplotlib.pyplot as plt
                import scipy.io.wavfile as wav
                import librosa.display as ld  

                def plot_wav_mel(wav_paths, save_path="./test/waveform_mel.png"):
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

                wavs = ['./test/origin.wav',
                './test/time_align_shit.wav',
                './test/orig.wav',
                './test/noised.wav',]

                plot_wav_mel(wavs)



                raise ValueError
                sdr = calculate_sisdr(mixture,edited_waveform)
                print(sdr)

                source = source.squeeze(0)
                mixture = mixture.squeeze(0)
                sdr = calculate_sisdr(mixture,mixture)
                print(sdr)
                sdr = calculate_sisdr(source,mixture)
                print(sdr)


                sdr_no_sep = calculate_sdr(ref=source, est=mixture)

                if self.query == 'caption':
                    text = [caption]
                elif self.query == 'labels':
                    text = [labels]

                ts = config['transfer_strength']
                guid = config['guidance_scale']
                totalstep = config['ddim_steps']
                iteration = config['iteration']
                duration = 10.24
                mel, _, stft_c, _, _ = pcr.read_audio_file(mixture_path)

                current_mel = mel
                for iter in range(iteration):
                    waveform = aldm.edit_audio_with_ddim(
                        mel=current_mel,
                        text=str(f'Nothing but {text[0]}'),
                        duration=duration,
                        batch_size=1,
                        transfer_strength=ts,
                        guidance_scale=guid,
                        ddim_steps=totalstep,
                        clipping = False,
                    )
                    waveform = waveform[None, ...]
                    mel, _ = pcr.wav_feature_extraction(waveform, pad_stft=True)
                    current_mel = mel

                sep_segment = waveform[0, ...]

                # sep_segment = pcr.normalize_wav(sep_segment)
                # source = pcr.normalize_wav(source)

                assert sep_segment.shape == source.shape, f'{sep_segment.shape}, {source.shape}'
                
                print(text[0])
                t = text[0]
                text = re.sub(r'[\/:*?"<>| ]', '_', t)

                sf.write(f'./z_result/{i}_result_{t}.wav', sep_segment, 16000)
                sf.write(f'./z_result/{i}_gt_{t}.wav', source, 16000)
                sf.write(f'./z_result/{i}_mixture_{t}.wav', mixture, 16000)

                sdr = calculate_sdr(ref=source, est=sep_segment)
                sdri = sdr - sdr_no_sep
                sisdr = calculate_sisdr(ref=source, est=sep_segment)

                #####
                sdr_no_sep_with_mix = calculate_sdr(ref=mixture, est=source)
                sdr_with_mix = calculate_sdr(ref=mixture, est=sep_segment)
                sdri_with_mix = sdr_with_mix - sdr_no_sep_with_mix
                sisdr_with_mix = calculate_sisdr(ref=mixture, est=sep_segment)
                sisdrs_list_with_mix.append(sisdr_with_mix)
                sdris_list_with_mix.append(sdri_with_mix)
                #####

                sisdrs_list.append(sisdr)
                sdris_list.append(sdri)

        # except Exception as e:
        #     print(f"Error: {e}")
        #     # traceback.print_exc()

        # finally:
        #     mean_sisdr = np.mean(sisdrs_list)
        #     mean_sdri = np.mean(sdris_list)
        #     mean_sisdr_with_mix = np.mean(sisdrs_list_with_mix)
        #     mean_sdri_with_mix = np.mean(sdris_list_with_mix)
            
        #     return mean_sisdr, mean_sdri, mean_sisdr_with_mix, mean_sdri_with_mix

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
    '''
    mel, _, _, _ = processor.read_audio_file(current_audio)
        # AudioLDM을 사용하여 스타일 변환 수행
        waveform = ldm.edit_audio_with_ddim(
            mel=mel,
            text=target_text,
            duration=10.24,
            batch_size=1,
            transfer_strength=transfer_strength,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            clipping = True,
        )
    '''

    config = {
        'transfer_strength': 0,
        'ddim_steps': 20,
        'guidance_scale': 2.5,
        'iteration': 1,
    }

    mean_sisdr, mean_sdri, with_mix_sisdr, with_mix_sdri = eval((processor, aldm), config)
    
    print(" SI-SDR  |  SDRi ")
    print(f"{round(mean_sisdr, 2)}  |  {round(mean_sdri, 2)}")
    print(" With Mix SI-SDR  |  With Mix SDRi ")
    print(f"{round(with_mix_sisdr, 2)}  |  {round(with_mix_sdri, 2)}")
    