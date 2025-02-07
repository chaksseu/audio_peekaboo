import os
import sys

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src_audioldm')
sys.path.extend([proj_dir, src_dir])

import soundfile as sf
import src_audioldm.audioldm as ldm
from audio_peekaboo.src_audioldm.utilities.data.dataprocessor import AudioDataProcessor

ldm = ldm.AudioLDM('cuda:0')
device = ldm.device


def iterative_audio_transform(ldm, audioprocessor, initial_audio, target_text, transfer_strength, num_iterations=5):
    current_audio = initial_audio  # 첫 번째 입력 오디오
    for i in range(1, num_iterations + 1):  # att2.wav ~ att{num_iterations+1}.wav
        output_audio = f"./att{i}.wav"
        
        # AudioLDM을 사용하여 스타일 변환 수행
        waveform = ldm.style_transfer(
            text=target_text,
            original_audio_file_path=current_audio,
            transfer_strength=transfer_strength,
            processor=audioprocessor,
            guidance_scale=1.5,
        )

        # 변환된 오디오 저장
        waveform = waveform.squeeze(0).detach().cpu().numpy()
        sf.write(output_audio, waveform, 16000)

        print(f"Generated: {output_audio}")
        
        # 다음 단계의 입력 오디오를 현재 출력으로 설정
        current_audio = output_audio


def run_peekaboo(target_text: str,):

    audioprocessor = AudioDataProcessor(device=device)

    # 예제 실행
    initial_audio_path = "./strength08_n5/a_cat_n_stepping_wood.wav"
    target_text = "A cat meowing"
    transfer_strength = 0.8
    num_iterations = 10  # 반복 횟수 (att2.wav ~ att6.wav 생성)

    iterative_audio_transform(ldm, audioprocessor, initial_audio_path, target_text, transfer_strength, num_iterations)

    raise ValueError

if __name__ == "__main__":

    run_peekaboo(
        target_text='A cat meowing',
        )
