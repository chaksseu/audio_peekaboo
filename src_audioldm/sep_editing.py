import os
import sys

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src_audioldm')
sys.path.extend([proj_dir, src_dir])

import soundfile as sf
import src_audioldm.audioldm as ldm
from src_audioldm.utilities.data.dataprocessor import AudioDataProcessor


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
        )
        if i % 1 == 0:
            # 변환된 오디오 저장
            sf.write(output_audio, waveform, 16000)
            print(f"Generated: {output_audio}")
        
        # 다음 단계의 입력 오디오를 현재 출력으로 설정
        current_audio = output_audio

if __name__ == "__main__":

    aldm = ldm.AudioLDM('cuda:0')
    device = aldm.device
    processor = AudioDataProcessor(device=device)

    # set1 = processor.making_dataset("./best_samples/Footsteps_on_a_wooden_floor.wav")
    # set2 = processor.making_dataset("./best_samples/Techno_music_with_a_strong__upbeat_tempo_and_high_melodic_riffs.wav")
    
    # mixed1, mixed2 = processor.get_mixed_sets(set1, set2, snr_db=3)
    
    # audio = set1['waveform'].squeeze(0).squeeze(0).detach().cpu().float().numpy()

    # sf.write('./birds.wav', audio, 16000)
    # audio = set2['waveform'].squeeze(0).squeeze(0).detach().cpu().float().numpy()

    # sf.write('./Techno.wav', audio, 16000)

    # audio = mixed2['waveform'].squeeze(0).squeeze(0).detach().cpu().float().numpy()

    # sf.write('./Footsteps_n_techno.wav', audio, 16000)

    # 예제 실행
    initial_audio_path = "./a_cat_n_stepping_wood.wav"
    # name = 'A_cat_meowing'
    name = "A_cat_meowing"
    target_text = name.replace('_',' ')
    print(target_text)
    # target_text = "A cat meowing"

    iterative_audio_transform(
        ldm=aldm, processor=processor,
        initial_audio=initial_audio_path,
        target_text=target_text,
        ddim_steps=20,
        transfer_strength=0.7,
        num_iterations=5,
        guidance_scale=2.5,
        )
