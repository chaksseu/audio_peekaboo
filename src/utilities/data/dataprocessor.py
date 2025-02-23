import os
import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn

# import src_audioldm.utilities.audio as Audio
"""
self.STFT = Audio.stft.TacotronSTFT(
            self.filter_length,
            self.hop_length,
            self.win_length,
            self.n_mel,
            self.sampling_rate,
            self.mel_fmin,
            self.mel_fmax,)
"""

def spectral_normalize_torch(magnitudes, C=1, CLIP_VAL=1e-5):  # dynamic_range_compression_torch
    return torch.log(torch.clamp(magnitudes, min=CLIP_VAL) * C)

class AudioDataProcessor():
    def __init__(self, device="cuda"):
        self.device = device

        self.pad_wav_start_sample = 0
        self.do_trim_wav = False
        self.waveform_only = False
        self.do_random_segment = False

        self.sampling_rate = 16000
        self.duration = 10.24
        self.target_length = 1024
        self.mixup = 0.0

        self.mel_basis = {}
        self.hann_window = {}

        # DSP: s-full 기준 (audioldm_original.yaml)
        self.filter_length = 1024  # n_fft
        self.hop_length = 160
        self.win_length = 1024
        self.n_mel = 64  # M: 64
        self.mel_fmin = 0
        self.mel_fmax = 8000

        self.n_freq = self.filter_length // 2 + 1  # F: 513
        self.sample_length = self.sampling_rate * self.duration  # N: 163840
        self.pad_size = int((self.filter_length - self.hop_length) / 2)  # (1024-160)/2 = 432
        self.n_times = int(((self.sample_length + 2 * self.pad_size) - self.win_length) // self.hop_length +1)  # 123

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

    def read_wav_file(self, filename):  # audiofile > mono ch > resample > norm > pad > norm => np[1, N:163840]
        # 1. 파일 로드
        waveform, original_sr = torchaudio.load(filename, normalize=True)  # ts[C,original_samples]
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform  # mono 변환 / ts[B,N]
        target_samples = int(original_sr * self.duration)  # original samples 길이
        # 2. random segment 추출 (target samples를 충족하는 선에서)
        random_start = None
        if self.do_random_segment:
            waveform, random_start = self.random_segment_wav(waveform, target_samples)
        # 3. resampling (설정한 sr에 맞게 변환)
        waveform = torchaudio.functional.resample(waveform, original_sr, self.sampling_rate)  # ts[1,target_samples]
        # 4. 전처리 단계
        waveform = waveform.numpy()[0, ...]  # numpy 변환 & 1st channel 선택 / np[target_samples,]
        waveform = self.normalize_wav(waveform)  # centering & Norm [-0.5,0.5]
        if self.do_trim_wav:
            waveform = self.trim_wav_(waveform)  # 무음 구간 제거
        # 5. 최종 형태로 변환
        waveform = waveform[None, ...]  # channel dim 추가 / np[1,target_samples]
        target_length = int(self.sampling_rate * self.duration)  # 최종 target samples 길이
        waveform = self.pad_wav(waveform, target_length)  # padding if wav is short
        waveform = self.normalize_wav(waveform)  #! github main code에서는 한번 더 Norm 했음
        return waveform, random_start  # np[1,target_samples], int

    # --------------------------------------------------------------------------------------------- #

    def waveform_to_stft(self, waveform):  # [1, N:163840] → [1, F:513, T:1024]
        
        assert torch.min(waveform) >= -1, f"train min value is {torch.min(waveform)}"
        assert torch.max(waveform) <= 1, f"train min value is {torch.max(waveform)}"

        if self.mel_fmax not in self.mel_basis:
            mel_filterbank = librosa_mel_fn(  # np[M:64, F:513]
                sr=self.sampling_rate,
                n_fft=self.filter_length,
                n_mels=self.n_mel,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,)
            
            self.mel_basis[f"{self.mel_fmax}_{self.device}"] = torch.from_numpy(mel_filterbank).float().to(self.device)  # ts[n_mel, n_freq=n_fft//2+1]
            self.hann_window[f"{self.device}"] = torch.hann_window(self.win_length).to(waveform.device)  # ts[win_length,] = [1024,]

        # ========== wav -> stft ==========
        pad_size = int((self.filter_length - self.hop_length) / 2)  # (1024-160)/2 = 432
        # waveform: np[C, samples] → [C, 1, samples] → [C, 1, samples + 2*pad_size] → [C, samples + 2*pad_size] = ts[C, 164704]
        waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (pad_size, pad_size), mode="reflect").squeeze(1)

        stft_complex = torch.stft(  # ts[1, F:513, T:1024~30] (complex)
            waveform,                   # F = filter_length // 2 + 1 (onesided=True) = 513
            self.filter_length,         # T = ((samples + 2*pad_size) - win_length) // hop_length + 1 = 1024
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[f"{self.device}"],
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        stft_mag = torch.abs(stft_complex)  # ts[1, F:513, T:1024~30]
        
        assert stft_complex.shape == stft_mag.shape
        assert stft_mag.shape[1] == self.n_freq, f"{stft_mag.shape}, {self.n_freq}, {self.n_times}"
        return stft_mag, stft_complex
    
    def stft_to_mel(self, stft_mag, stft_complex):
        # ========== stft -> mel ==========
        mel_filterbank = self.mel_basis[f"{self.mel_fmax}_{self.device}"]  # ts[M:64, F:513]
        # [M:64, F:513] x [1, F:513, T:1024~] → [1, M:64, T:1024~]
        stft_mag = stft_mag.to(self.device)  # ts[1, F:513, T:1024~]
        mel_spec = spectral_normalize_torch(torch.matmul(mel_filterbank, stft_mag))  # ts[1, M:64, T:1024~]

        assert mel_spec.shape[1] == self.n_mel and stft_mag.shape[1] == stft_complex.shape[1] == self.n_freq, f"{mel_spec.shape}, {stft_mag.shape}, {stft_complex.shape}"
        return mel_spec, stft_mag, stft_complex  # ts[1, M:64, T:1024~] / ts[1, F:513, T:1024~] / ts[1, F:513, T:1024~]
    
    def pad_spec(self, spectrogram, do_pad):  # [T, ~] → [T*, ~*]
        n_frames = spectrogram.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            spectrogram = m(spectrogram)  # [T*, ~] 뒷 시간 늘림
        elif p < 0:
            spectrogram = spectrogram[0 : self.target_length, :]  # [T*, ~] 뒷 시간 줄임
        if (spectrogram.size(-1) % 2 != 0) and do_pad:
            spectrogram = spectrogram[..., :-1]  # ~ 가 odd면, -1
        return spectrogram, p

    def postprocess_spec(self, spectrogram, do_pad=True):  # [1, ~, T] -> [T*, ~*]
        spec = spectrogram[0]  # [~, T]
        spec = spec.T.float()  # [T, ~]
        spec, p = self.pad_spec(spec, do_pad)  # [T*, ~*]
        return spec, p

    def reversing_stft(self, stft):
        if len(stft.shape) == 3:
            stft = stft.squeeze(0)
        assert stft.shape == torch.Size([self.n_times, self.n_freq]), f"{stft.shape}"
        stft = stft.T.float()
        stft = stft.unsqueeze(0)
        return stft

    def wav_feature_extraction(self, waveform, pad_stft=False):  # wav: np[C,N] → logmel: ts[1,1,T,M] / stft: ts[1,1,T,F]
        waveform = waveform[0, ...]  # 다채널 방지 / np[samples,] = (163840,)
        waveform = torch.FloatTensor(waveform).unsqueeze(0).to(self.device)  # ts[1, samples]
        stft, stft_c = self.waveform_to_stft(waveform)  # ts[1, F:513, T:1024~] / ts[1, F:513, T:1024~]
        log_mel_spec, stft, stft_c = self.stft_to_mel(stft, stft_c)  # ts[1, M:64, T:1024~] / ts[1, F:513, T:1024~] / ts[1, F:513, T:1024~]
        log_mel_spec, p = self.postprocess_spec(log_mel_spec)  # ts[T:1024, M:64]
        stft, p = self.postprocess_spec(stft, do_pad=pad_stft)  # ts[T:1024, F:512]

        return log_mel_spec[None, None, ...], stft[None, None, ...], stft_c  # ts[1,1,T,M] / ts[1,1,T,F] / ts[1, F:513, T:1024~]

    # --------------------------------------------------------------------------------------------- #

    def read_audio_file(self, filename, pad_stft=False):  # → ts[t,mel], ts[t,freq], ts[C,samples]
        # 1. 오디오 파일 로드 또는 빈 파형 생성
        if os.path.exists(filename):
            waveform, random_start = self.read_wav_file(filename)  # np[C,samples], int
        else:
            target_length = int(self.sampling_rate * self.duration)
            waveform, random_start = torch.zeros((1, target_length)), 0  # np[C,samples], int
            print(f'Non-fatal Warning [dataset.py]: The wav path "{filename}" not found. Using empty waveform.')
        waveform = torch.FloatTensor(waveform)

        # 2. 특성 추출 (stft spec, log mel spec)
        log_mel_spec, stft, stft_c = (None, None, None) if self.waveform_only else self.wav_feature_extraction(waveform, pad_stft=pad_stft)  # input: [1,N]
        return log_mel_spec, stft, stft_c, waveform, random_start  # ts[1,1,T,M] / ts[1,1,T,F] / ts[1, F:513, T:1024~] / ts[1,N]

    # --------------------------------------------------------------------------------------------- #

    def inverse_mel_with_phase(
        self,
        masked_mel_spec: torch.Tensor,    # 모델이 예측한 log mel spec, shape [B, T, n_mel]
        stft_complex: torch.Tensor,       # forward에서 구한 복소 STFT, shape [B, n_freq, n_time]
        mel_filterbank: torch.Tensor=None,     # shape [n_mel, n_freq], forward에서 쓴 것과 동일
        hann_window: torch.Tensor=None,        # forward와 동일한 Hann window
        filter_length=1024,
        hop_length=160,
        win_length=1024,
        pad_size=432,
        device="cuda:0",
        eps=1e-5
    ):
        """
        masked_mel_spec (log scale) + stft_complex(phase) → 근사 waveform
        """
        # 1) mel_spec (log scale) → linear scale로 변환
        #    forward에서 spectral_normalize_torch = log(clip(mag)*C) 사용했으므로
        #    여기서는 exp()로 복원 (C=1 가정)
        assert masked_mel_spec.shape == (1,1,1024,64)
        masked_mel_spec = masked_mel_spec.squeeze(0)
        masked_mel_linear = torch.exp(masked_mel_spec)  # shape [1, T, M]

        # 2) mel → STFT magnitude로 근사 복원
        #    mel_filterbank shape이 [n_mel, n_freq]이므로 pseudo-inverse를 구함
        #    (n_mel=64 << n_freq=513 이라 완벽 역변환은 불가)
        if not mel_filterbank:
            mel_filterbank = self.mel_basis[f"{self.mel_fmax}_{self.device}"]
            inv_mel_filter = torch.pinverse(mel_filterbank)  # shape [F, M]

        # 현재 masked_mel_linear: [B, T, n_mel] → [B, n_mel, T] 로 transpose
        masked_mel_linear = masked_mel_linear.permute(0, 2, 1)  # shape [B, M, T]

        # pseudo-inverse 곱: [F, M] x [M, T] = [F, T]
        # 배치처리까지 고려하려면 map으로 처리
        batch_size = masked_mel_linear.shape[0]
        masked_stft_mag = []
        for i in range(batch_size):
            # shape [n_mel, T] → [n_freq, T]
            mag_i = inv_mel_filter @ masked_mel_linear[i]
            masked_stft_mag.append(mag_i.unsqueeze(0))
        masked_stft_mag = torch.cat(masked_stft_mag, dim=0)  # shape [B, F, T]

        # 3) 원본 stft_complex의 phase 추출 후, magnitude와 결합
        #    phase = stft_complex / (|stft_complex| + eps)
        phase = stft_complex / (stft_complex.abs() + eps)
        masked_stft_complex = masked_stft_mag * phase  # shape 동일: [B, F, T]

        # 4) iSTFT 수행 (forward와 동일 파라미터)
        #    center=False이므로, forward 시 (pad_size, pad_size) reflect padding 했었음.
        #    여기서도 그대로 동일 파라미터 유지
        if not hann_window:
            hann_window = self.hann_window[f"{self.device}"]

        estimated_wav = torch.istft(
            masked_stft_complex.to(self.device),
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=hann_window,
            normalized=False,
            onesided=True
        )  # shape [B, samples + 2*pad_size]

        # 5) pad_size 부분 제거
        #    forward에서 reflect pad를 (pad_size, pad_size)만큼 했었으므로,
        #    최종 waveform에서 앞뒤로 pad_size samples씩 잘라낸다.
        if estimated_wav.shape[-1] > pad_size*2:
            estimated_wav = estimated_wav[..., pad_size:-pad_size]  # shape [B, samples]
        else:
            # 혹시 길이가 매우 짧다면 예외처리
            estimated_wav = estimated_wav[..., 0:1]

        return estimated_wav  # shape [B, samples]


    # --------------------------------------------------------------------------------------------- #

    def making_dataset(self, file_path):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        text = filename.replace('_', ' ')
        print(text)
        if text is None:
            print("Warning: The model return None on key text", filename); text = ""

        log_mel_spec, stft, stft_c, waveform, random_start = self.read_audio_file(file_path)
        
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
        assert data["stft"].shape == torch.Size([1, 1, 1024, 512]), data["stft"].shape
        assert data["log_mel_spec"].shape == torch.Size([1, 1, 1024, 64]), data["log_mel_spec"].shape
        return data

    def get_mixed_sets(self, set1, set2, snr_db=0):
        wav1, wav2 = set1["waveform"], set2["waveform"]  # ts[1,1,samples]
        assert wav1.shape == wav2.shape
        power1, power2 = torch.mean(wav1 ** 2), torch.mean(wav2 ** 2)
        print(power1, power2)
        scaling_factor = torch.sqrt(power1 / (power2 + 1e-7) * 10 ** (-snr_db/10))
        mixed = wav1 + wav2 * scaling_factor
        max_abs = mixed.abs().max()
        if max_abs > 1:
            mixed /= max_abs
        mixed_wav = (mixed * 0.5).float().squeeze(0)  # ts[1,1,samples]
        return mixed_wav

        # log_mel_spec, stft, c = self.waveform_to_mel_n_stft(mixed_wav)
        # log_mel_spec = self.postprocess_spec(log_mel_spec).unsqueeze(0).float()  # [1,t,mel]
        # stft = self.postprocess_spec(stft, do_pad=False).unsqueeze(0).float()  # [1,t,freq]
        # def set_dict(batch):
        #     return {
        #         "text": batch["text"],                        # List, [1]
        #         "fname": batch["fname"],                      # List, [1]
        #         "waveform": mixed_wav.unsqueeze(0).to(self.device),        # Tensor, [1, 1, samples_num]
        #         "stft": stft.to(self.device),                 # Tensor, [1, t-steps, f-bins]
        #         "log_mel_spec": log_mel_spec.to(self.device), # Tensor, [1, t-steps, mel-bins]
        #     }
        # mixed_set1 = set_dict(set1)
        # mixed_set2 = set_dict(set2)
        # assert mixed_set1["waveform"].shape == torch.Size([1, 1, 163840]), mixed_set1["waveform"].shape
        # assert mixed_set1["stft"].shape[:-1] == torch.Size([1, 1, 1024, 512]), mixed_set1["stft"].shape
        # assert mixed_set1["log_mel_spec"].shape == torch.Size([1, 1, 1024, 64]), mixed_set1["log_mel_spec"].shape

        # return (mixed_set1, mixed_set2)
