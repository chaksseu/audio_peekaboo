import os
import sys

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src_audioldm')
sys.path.extend([proj_dir, src_dir])

import soundfile as sf
from src.audioldm import AudioLDM as ldm
from src.utilities.data.dataprocessor import AudioDataProcessor as prcssr


def iterative_audio_transform(
        ldm, processor,
        initial_audio,
        target_text,
        transfer_strength,
        guidance_scale,
        ddim_steps,
        num_iterations=5,
        ):
    current_audio = initial_audio  # 첫 번째 입력 오디오
    for i in range(1, num_iterations + 1):
        output_audio = f"./att{i}.wav"
        
        mel, _, _, _, _ = processor.read_audio_file(current_audio)
        # AudioLDM을 사용하여 스타일 변환 수행
        waveform = ldm.edit_audio_with_ddim(
            mel=mel,      ###
            text=target_text,
            duration=10.24,
            batch_size=1,
            transfer_strength=transfer_strength,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            clipping = False,
        )
        if i % 1 == 0:
            # 변환된 오디오 저장       ###

            waveform = waveform.detach().cpu().numpy().squeeze(0)
            sf.write(output_audio, waveform, 16000)
            print(f"Generated: {output_audio}")

        
        
        # 다음 단계의 입력 오디오를 현재 출력으로 설정
        current_audio = output_audio

if __name__ == "__main__":

    aldm = ldm('cuda:0')
    device = aldm.device
    pcr = prcssr(device=device)

    # # set1 = processor.making_dataset('./samples/best_samples/A_cat_meowing.wav')
    # # set2 = processor.making_dataset('./samples/best_samples/Footsteps_on_a_wooden_floor.wav')

    # # mixed_wav = processor.get_mixed_sets(set1, set2, snr_db=0)
    # # mixed_wav = mixed_wav.squeeze(0).squeeze(0)
    # # sf.write('./a_cat_n_stepping_wood.wav', mixed_wav, 16000)


    # # setting
    # initial_audio_path = "./a_cat_n_stepping_wood.wav"
    # name = "A_cat_meowing"
    # target_text = name.replace('_',' ')
    # print(target_text)
    # # target_text = "A cat meowing"

    # iterative_audio_transform(
    #     ldm=aldm, processor=processor,
    #     initial_audio=initial_audio_path,
    #     target_text=target_text,
    #     ddim_steps=200,
    #     transfer_strength=0.4,
    #     num_iterations=5,
    #     guidance_scale=2.5,
    #     )

    from evaluation.evaluate_audiocaps import calculate_sisdr

    mel, _, stft_complex, wav_pri, rand_start = pcr.read_audio_file('./samples/best_samples/A_cat_meowing.wav')  # np[1,N]
    mel, _, stft_complex, wav_sep, rand_start = pcr.read_audio_file('./att4.wav')  # np[1,N]
    wav_pri = wav_pri.cpu().numpy().squeeze(0)  # [:,:leng]
    wav_sep = wav_sep.cpu().numpy().squeeze(0)

    sdr = calculate_sisdr(wav_pri,wav_sep)
    print(sdr)

    import numpy as np
    from scipy.signal import correlate

    def align_waveforms(wav1: np.ndarray, wav2: np.ndarray):
        # Cross-correlation을 이용해 최적의 shift 찾기
        corr = correlate(wav1, wav2, mode='full')
        shift = np.argmax(corr) - len(wav2) + 1  # wav1이 wav2보다 얼마나 밀려 있는지

        # Shift 적용하여 wave 정렬
        if shift > 0:
            wav1 = wav1[shift:]  # wav1을 오른쪽으로 이동
        elif shift < 0:
            wav2 = wav2[-shift:]  # wav2를 오른쪽으로 이동

        # 겹치는 부분만 남기기
        print(shift)
        min_len = min(len(wav1), len(wav2))
        return wav1[:min_len], wav2[:min_len]
    
    wav, wav_ = align_waveforms(wav_pri, wav_sep)
    print(len(wav_))
    sdr = calculate_sisdr(wav,wav_)
    print(sdr)



