import os
import sys
import re
from typing import Dict, List

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

        with torch.no_grad():
            for eval_data in tqdm(self.eval_list):

                idx, caption, labels, _, _ = eval_data

                source_path = os.path.join(self.audio_dir, f'segment-{idx}.wav')
                mixture_path = os.path.join(self.audio_dir, f'mixture-{idx}.wav')

                source, fs = librosa.load(source_path, sr=self.sampling_rate, mono=True)
                mixture, fs = librosa.load(mixture_path, sr=self.sampling_rate, mono=True)

                sdr_no_sep = calculate_sdr(ref=source, est=mixture)
                                
                if self.query == 'caption':
                    text = [caption]
                elif self.query == 'labels':
                    text = [labels]

                # conditions = pl_model.query_encoder.get_query_embed(
                #     modality='text',
                #     text=text,
                #     device=device)
                # input_dict = {
                #     "mixture": torch.Tensor(mixture)[None, None, :].to(device),
                #     "condition": conditions,}

                # sep_segment = pl_model.ss_model(input_dict)["waveform"]
                #     # sep_segment: (batch_size=1, channels_num=1, segment_samples)

                # sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
                #     # sep_segment: (segment_samples,)

                ts, guid, totalstep = config
                mel, _, _, _ = pcr.read_audio_file(mixture_path)
                sep_segment = aldm.edit_audio_with_ddim(
                    mel=mel,
                    text=text[0],
                    duration=10.24,
                    batch_size=1,
                    transfer_strength=ts,
                    guidance_scale=guid,
                    ddim_steps=totalstep,
                    clipping = True,
                )

                assert sep_segment.shape == source

                sdr = calculate_sdr(ref=source, est=sep_segment)
                sdri = sdr - sdr_no_sep
                sisdr = calculate_sisdr(ref=source, est=sep_segment)

                sisdrs_list.append(sisdr)
                sdris_list.append(sdri)

        mean_sisdr = np.mean(sisdrs_list)
        mean_sdri = np.mean(sdris_list)
        
        return mean_sisdr, mean_sdri

if __name__ == "__main__":
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
    eval()