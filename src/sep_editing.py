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
        if i % 1 == 0:
            # 변환된 오디오 저장
            sf.write(output_audio, waveform, 16000)
            print(f"Generated: {output_audio}")
        
        # 다음 단계의 입력 오디오를 현재 출력으로 설정
        current_audio = output_audio

if __name__ == "__main__":

    aldm = ldm('cuda:0')
    device = aldm.device
    processor = prcssr(device=device)

    set1 = processor.making_dataset('./samples/best_samples/A_cat_meowing.wav')
    set2 = processor.making_dataset('./samples/best_samples/Footsteps_on_a_wooden_floor.wav')

    mixed_wav = processor.get_mixed_sets(set1, set2, snr_db=0)
    mixed_wav = mixed_wav.squeeze(0).squeeze(0)
    sf.write('./a_cat_n_stepping_wood.wav', mixed_wav, 16000)


    # # setting
    # initial_audio_path = "./a_cat_n_stepping_wood.wav"
    # # name = 'A_cat_meowing'
    # name = "A_cat_meowing"
    # target_text = name.replace('_',' ')
    # print(target_text)
    # # target_text = "A cat meowing"

    # iterative_audio_transform(
    #     ldm=aldm, processor=processor,
    #     initial_audio=initial_audio_path,
    #     target_text=target_text,
    #     ddim_steps=25,
    #     transfer_strength=0.15,
    #     num_iterations=10,
    #     guidance_scale=2.5,
    #     )
