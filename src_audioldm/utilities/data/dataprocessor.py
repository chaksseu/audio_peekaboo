import os
import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn

# import src_audioldm.utilities.audio as Audio

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

class AudioDataProcessor():
    def __init__(self, device="cuda"):
        self.device = device

        self.pad_wav_start_sample = 0
        self.trim_wav = False
        self.waveform_only = False

        self.melbins = 64     # 중복
        self.sampling_rate = 16000  # 중복
        self.hopsize = 160    # 중복
        self.duration = 10.24
        self.target_length = 1024
        self.mixup = 0.0

        self.mel_basis = {}
        self.hann_window = {}

        # DSP: s-full 기준 (audioldm_original.yaml)
        self.filter_length = 1024
        self.hop_length = 160
        self.win_length = 1024
        self.n_mel = 64
        self.mel_fmin = 0
        self.mel_fmax = 8000

        self.n_freq = self.filter_length // 2 + 1
        self.sample_length = self.sampling_rate * self.duration
        self.pad_size = int((self.filter_length - self.hop_length) / 2)
        self.n_times = ((self.sample_length + 2 * self.pad_size) - self.win_length // self.hop_length +1)

        # self.STFT = Audio.stft.TacotronSTFT(
        #             self.filter_length,
        #             self.hop_length,
        #             self.win_length,
        #             self.n_mel,
        #             self.sampling_rate,
        #             self.mel_fmin,
        #             self.mel_fmax,)

    # --------------------------------------------------------------------------------------------- #

    def random_segment_wav(self, waveform, target_length):  # target sample 길이에 맞게 random 추출
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, f"Waveform is too short, {waveform_length}"
        # Too short
        if waveform_length <= target_length:
            return waveform, 0
        # 10번 시도에도 적절한 세그먼트 못찾은 경우, 마지막 시도 반환
        for _ in range(10):
            random_start = int(self.random_uniform(0, waveform_length - target_length))
            segment = waveform[:, random_start:random_start + target_length]
            if torch.max(torch.abs(segment)) > 1e-4:
                return segment, random_start
        return segment, random_start

    def normalize_wav(self, waveform):  # waveform Normalizing
        MAX_AMPLITUDE = 0.5  # Max amplitude를 0.5로 manually하게 limit 둠
        EPSILON = 1e-8
        centered = waveform - np.mean(waveform)
        normalized = centered / (np.max(np.abs(centered)) + EPSILON)  # in [-1,1] 
        return normalized * MAX_AMPLITUDE    # in [-0.5,0.5]

    def trim_wav_(self, waveform, threshold=0.0001, chunk_size=1000):  # wav 시작&끝의 무음 구간을 제거하는(trim) 함수
        if np.max(np.abs(waveform)) < threshold:
            return waveform
        def find_sound_boundary(samples, reverse=False):
            length = samples.shape[0]
            pos, limit, step = (length, 0, -chunk_size) if reverse else (0, length, chunk_size)
            
            while ((pos - step) if reverse else (pos + chunk_size)) > limit:
                chunk_start = (pos - chunk_size) if reverse else pos
                chunk_end = pos if reverse else (pos + chunk_size)
                if np.max(np.abs(samples[chunk_start:chunk_end])) < threshold:
                    pos += step
                else:
                    break
            return pos + (chunk_size if reverse else 0)
        start = find_sound_boundary(waveform, reverse=False)
        end = find_sound_boundary(waveform, reverse=True)
        return waveform[start:end]

    def random_uniform(self, start, end):  # 주어진 범위 내에서 uniform scalar sampling
        val = torch.rand(1).item()
        return start + (end - start) * val

    def pad_wav(self, waveform, target_length):  # wav를 목표 길이로 padding -> padded_wav
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, f"Waveform is too short, {waveform_length}"

        if waveform_length == target_length:
            return waveform
        # Padding (target length가 waveform length보다 더 긴 경우만 처리하면 됨)
        padded_wav = np.zeros((1, target_length), dtype=np.float32)
        random_start = int(self.random_uniform(0, target_length - waveform_length))
        start_pos = 0 if self.pad_wav_start_sample else random_start
        
        padded_wav[:, start_pos:start_pos + waveform_length] = waveform
        return padded_wav

    def read_wav_file(self, filename):  # 오디오 파일을 읽고 전처리
        # 1. 파일 로드
        waveform, original_sr = torchaudio.load(filename)  # ts[C,original_samples]
        target_samples = int(original_sr * self.duration)  # original samples 길이
        # 2. random segment 추출 (target samples를 충족하는 선에서)
        waveform, random_start = self.random_segment_wav(waveform, target_samples)
        # 3. resampling (설정한 sr에 맞게 변환)
        waveform = torchaudio.functional.resample(waveform, original_sr, self.sampling_rate)  # ts[C,target_samples]
        # 4. 전처리 단계
        waveform = waveform.numpy()[0, ...]  # numpy 변환 & 1st channel 선택 / np[target_samples,]
        waveform = self.normalize_wav(waveform)  # centering & Norm [-0.5,0.5]
        if self.trim_wav:
            waveform = self.trim_wav_(waveform)  # 무음 구간 제거
        # 5. 최종 형태로 변환
        waveform = waveform[None, ...]  # channel dim 추가 / np[C,target_samples]
        target_length = int(self.sampling_rate * self.duration)  # 최종 target samples 길이
        waveform = self.pad_wav(waveform, target_length)  # padding if wav is short
        # waveform = self.normalize_wav(waveform)  #! github main code에서는 한번 더 Norm 했음
        return waveform, random_start  # np[C,target_samples], int

    # --------------------------------------------------------------------------------------------- #

    def waveform_to_stft(self, waveform):  # [C,N] → [C,freq,t]
        # waveform: ts[C,samples] = [C,163840] in [-1,1]
        assert torch.min(waveform) >= -1, f"train min value is {torch.min(waveform)}"
        assert torch.max(waveform) <= 1, f"train min value is {torch.max(waveform)}"

        if self.mel_fmax not in self.mel_basis:
            mel_filterbank = librosa_mel_fn(
                sr=self.sampling_rate,
                n_fft=self.filter_length,
                n_mels=self.n_mel,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,)  # np[n_mel, n_freq=n_fft//2+1] = [64,513]
            
            self.mel_basis[f"{self.mel_fmax}_{self.device}"] = torch.from_numpy(mel_filterbank).float().to(self.device)  # ts[n_mel, n_freq=n_fft//2+1]
            self.hann_window[f"{self.device}"] = torch.hann_window(self.win_length).to(waveform.device)  # ts[win_length,] = [1024,]

        pad_size = int((self.filter_length - self.hop_length) / 2)  # (1024-160)/2 = 432
        # waveform: [C, samples] → [C, 1, samples] → [C, 1, samples + 2*pad_size] → [C, samples + 2*pad_size] = ts[C,164704]
        waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (pad_size, pad_size), mode="reflect").squeeze(1)

        stft_complex = torch.stft(
            waveform,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[f"{self.device}"],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )  # [C, n_freq, n_time] (complex) = ts[1,513,1024]
        # n_freq = filter_length // 2 + 1 (onesided=True) = 513
        # n_time = ((samples + 2*pad_size) - win_length) // hop_length + 1 = 1024

        stft_mag = torch.abs(stft_complex)  # ts[1,513,1024]
        
        return stft_mag  # [C,freq,t]
    
    def stft_to_mel(self, stft):  # [C,freq,t] → [C,mel,t]
        if len(stft.shape) != 3:
            stft = self.reversing_stft(stft)
        assert stft.shape[-2:] == [self.n_freq, self.n_times], f"{stft.shape}"
        
        mel_filterbank = self.mel_basis[f"{self.mel_fmax}_{self.device}"]  # ts[64,513]
        # [n_mel, n_freq] x [C, n_freq, n_time] → [C, n_mel, n_time] = [C,64,1024]
        mel_spec = spectral_normalize_torch(torch.matmul(mel_filterbank, stft))

        return mel_spec  # [C,mel,t]
    
    def waveform_to_mel_n_stft(self, waveform):  # ts[C,N] → logmel: [C,mel,t] / stft: [C,freq,t]
        "process: waveform → STFT → mel 변환 → normalize"
        stft_mag = self.waveform_to_stft(waveform)  # ts[C, n_freq, n_time] = [1,513,1024]
        mel_spec = self.stft_to_mel(stft_mag)  # ts[C,n_mel,n_time] = [1,64,1024]

        return mel_spec, stft_mag

    def pad_spec(self, spectrogram):  # [t,-] → [t,*]
        n_frames = spectrogram.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            spectrogram = m(spectrogram)
        elif p < 0:
            spectrogram = spectrogram[0 : self.target_length, :]

        if spectrogram.size(-1) % 2 != 0:
            spectrogram = spectrogram[..., :-1]

        return spectrogram

    def postprocess_spec(self, spectrogram, do_pad=True):  # [C,-,t]
        spec = spectrogram[0]  # [-,t]
        spec = spec.T.float()  # [t,-]
        if do_pad:
            spec = self.pad_spec(spec)  # [t,*]
        return spec

    def reversing_stft(self, stft):
        if len(stft.shape) == 3:
            stft = stft.squeeze(0)
        assert stft.shape == [self.n_times, self.n_freq], f"{stft.shape}"
        stft = stft.T.float()
        stft = stft.unsqueeze(0)
        return stft

    def wav_feature_extraction(self, waveform, pad_stft=False):  # wav: np[C,N] → logmel: [t,mel] / stft: [t,freq]
        waveform = waveform[0, ...]  # 다채널 방지 / np[samples,] = (163840,)
        waveform = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)  # ts[1, samples]
        log_mel_spec, stft = self.waveform_to_mel_n_stft(waveform)  # [C,mel,t] / [C,freq,t]
        log_mel_spec = self.postprocess_spec(log_mel_spec)  # [t,mel]
        stft = self.postprocess_spec(stft, do_pad=pad_stft)  # [t,freq]

        return log_mel_spec, stft  # [t,mel], [t,freq]

    # --------------------------------------------------------------------------------------------- #

    def read_audio_file(self, filename):  # → ts[t,mel], ts[t,freq], ts[C,samples]
        # 1. 오디오 파일 로드 또는 빈 파형 생성
        if os.path.exists(filename):
            waveform, random_start = self.read_wav_file(filename)  # np[C,samples], int
        else:
            target_length = int(self.sampling_rate * self.duration)
            waveform, random_start = torch.zeros((1, target_length)), 0  # np[C,samples], int
            print(f'Non-fatal Warning [dataset.py]: The wav path "{filename}" not found. Using empty waveform.')
        waveform = torch.FloatTensor(waveform)

        # 2. 특성 추출 (stft spec, log mel spec)
        log_mel_spec, stft = (None, None) if self.waveform_only else self.wav_feature_extraction(waveform, pad_stft=True)
        return log_mel_spec, stft, waveform, random_start  # ts[t,mel], ts[t,freq], ts[C,samples]

    def making_dataset(self, file_path):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        text = filename.replace('_', ' ')
        print(text)
        if text is None:
            print("Warning: The model return None on key text", filename); text = ""

        log_mel_spec, stft, waveform, random_start = self.read_audio_file(file_path)
        
        # return 할때 기준 shape
        data = {  
            "text": [text],                                                          # list[B]
            "fname": [filename],                                                     # list[B]
            "waveform": "" if (waveform is None) else waveform.float(),              # ts[B,1,samples_num]
            "stft": "" if (stft is None) else stft.float(),                          # ts[B,t,f]
            "log_mel_spec": "" if (log_mel_spec is None) else log_mel_spec.float(),  # ts[B,t,mel]
            }
        
        for key, value in data.items():
            if key in ["waveform", "stft", "log_mel_spec"]:
                data[key] = value.unsqueeze(0)

        assert data["waveform"].shape == torch.Size([1, 1, 163840]), data["waveform"].shape
        assert data["stft"].shape[:-1] == torch.Size([1, 1024]), data["stft"].shape
        assert data["log_mel_spec"].shape == torch.Size([1, 1024, 64]), data["log_mel_spec"].shape
        return data

    def get_mixed_sets(self, set1, set2, snr_db=0):
        wav1, wav2 = set1["waveform"], set2["waveform"]  # ts[1,1,samples]
        assert wav1.shape == wav2.shape
        power1, power2 = torch.mean(wav1 ** 2), torch.mean(wav2 ** 2)
        scaling_factor = torch.sqrt(power1 / power2 * 10 ** (-snr_db/10))
        mixed = wav1 + wav2 * scaling_factor
        max_abs = mixed.abs().max()
        if max_abs > 1:
            mixed /= max_abs
        mixed_wav = (mixed * 0.5).float().squeeze(0)  # ts[1,1,samples]

        log_mel_spec, stft = self.waveform_to_mel_n_stft(mixed_wav)
        log_mel_spec = self.postprocess_spec(log_mel_spec).unsqueeze(0).float()  # [1,t,mel]
        stft = self.postprocess_spec(stft, do_pad=False).unsqueeze(0).float()  # [1,t,freq]
        def set_dict(batch):
            return {
                "text": batch["text"],                        # List, [1]
                "fname": batch["fname"],                      # List, [1]
                "waveform": mixed_wav.to(self.device),        # Tensor, [1, 1, samples_num]
                "stft": stft.to(self.device),                 # Tensor, [1, t-steps, f-bins]
                "log_mel_spec": log_mel_spec.to(self.device), # Tensor, [1, t-steps, mel-bins]
            }
        mixed_set1 = set_dict(set1)
        mixed_set2 = set_dict(set2)
        assert mixed_set1["waveform"].shape == torch.Size([1, 1, 163840]), mixed_set1["waveform"].shape
        assert mixed_set1["stft"].shape[:-1] == torch.Size([1, 1024]), mixed_set1["stft"].shape
        assert mixed_set1["log_mel_spec"].shape == torch.Size([1, 1024, 64]), mixed_set1["log_mel_spec"].shape

        return (mixed_set1, mixed_set2)
