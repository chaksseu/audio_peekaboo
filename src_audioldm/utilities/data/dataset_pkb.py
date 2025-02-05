import os
import numpy as np
import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn



import src_audioldm.utilities.audio as Audio

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

        self.STFT = Audio.stft.TacotronSTFT(
                    self.filter_length,
                    self.hop_length,
                    self.win_length,
                    self.n_mel,
                    self.sampling_rate,
                    self.mel_fmin,
                    self.mel_fmax,)
        
    def random_segment_wav(self, waveform, target_length):
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

    def normalize_wav(self, waveform):  # 오디오 파형을 정규화
        MAX_AMPLITUDE = 0.5
        EPSILON = 1e-8
        centered = waveform - np.mean(waveform)
        normalized = centered / (np.max(np.abs(centered)) + EPSILON)
        return normalized * MAX_AMPLITUDE  # Manually limit the maximum amplitude into 0.5

    def trim_wav_(self, waveform, threshold=0.0001, chunk_size=1000):  # wav의 시작&끝에 있는 무음 구간을 제거하는(trim) 함수
        """      
        Args:
            waveform: 입력 오디오 파형
            threshold: 무음으로 간주할 진폭 임계값
            chunk_size: 한 번에 처리할 샘플 수
        """
        if np.max(np.abs(waveform)) < threshold:
            return waveform
        def find_sound_boundary(samples, reverse=False):
            length = samples.shape[0]
            pos = length if reverse else 0
            limit = 0 if reverse else length
            step = -chunk_size if reverse else chunk_size
            
            while (pos - step if reverse else pos + chunk_size) > limit:
                chunk_start = pos - chunk_size if reverse else pos
                chunk_end = pos if reverse else pos + chunk_size
                if np.max(np.abs(samples[chunk_start:chunk_end])) < threshold:
                    pos += step
                else:
                    break
            return pos + (chunk_size if reverse else 0)
        start = find_sound_boundary(waveform, reverse=False)
        end = find_sound_boundary(waveform, reverse=True)
        return waveform[start:end]

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def pad_wav(self, waveform, target_length):  # wav를 목표 길이로 padding -> padded_wav
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, f"Waveform is too short, {waveform_length}"

        if waveform_length == target_length:
            return waveform
        # Pad
        padded_wav = np.zeros((1, target_length), dtype=np.float32)
        start_pos = 0 if self.pad_wav_start_sample else int(self.random_uniform(0, target_length - waveform_length))
        
        padded_wav[:, start_pos:start_pos + waveform_length] = waveform
        return padded_wav

    def read_wav_file(self, filename):  # 오디오 파일을 읽고 전처리
        waveform, original_sr = torchaudio.load(filename)                            # 1. 파일 로드
        target_samples = int(original_sr * self.duration)                             # 2. 랜덤 세그먼트 추출
        waveform, random_start = self.random_segment_wav(waveform, target_samples)
        waveform = torchaudio.functional.resample(waveform, original_sr, self.sampling_rate)  # 3. 리샘플링
                                                                                        # 4. 전처리 단계
        waveform = waveform.numpy()[0, ...]                                           #     numpy 변환 및 첫 번째 채널 선택
        waveform = self.normalize_wav(waveform)                                       #     정규화
        if self.trim_wav:
            waveform = self.trim_wav_(waveform)                                        #     무음 구간 제거
                                                                                        # 5. 최종 형태로 변환
        waveform = waveform[None, ...]                                                #     채널 차원 추가
        target_length = int(self.sampling_rate * self.duration)
        waveform = self.pad_wav(waveform, target_length)                              #     패딩
        
        return waveform, random_start

    # --------------------------------------------------------------------------------------------- #

    def mel_spectrogram_train(self, wav_y):
        if torch.min(wav_y) < -1.0:
            print("train min value is ", torch.min(wav_y))
        if torch.max(wav_y) > 1.0:
            print("train max value is ", torch.max(wav_y))

        if self.mel_fmax not in self.mel_basis:
            mel = librosa_mel_fn(
                sr=self.sampling_rate,
                n_fft=self.filter_length,
                n_mels=self.n_mel,
                fmin=self.mel_fmin,
                fmax=self.mel_fmax,)
            
            self.mel_basis[f"{self.mel_fmax}_{wav_y.device}"] = torch.from_numpy(mel).float().to(wav_y.device)
            self.hann_window[f"{wav_y.device}"] = torch.hann_window(self.win_length).to(wav_y.device)

        pad_size = int((self.filter_length - self.hop_length) / 2)
        wav_y = torch.nn.functional.pad(wav_y.unsqueeze(1), (pad_size, pad_size), mode="reflect").squeeze(1)

        stft_spec = torch.stft(
            wav_y,
            self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[f"{wav_y.device}"],
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        stft_spec = torch.abs(stft_spec)
        
        mel_basis = self.mel_basis[f"{self.mel_fmax}_{wav_y.device}"]
        mel = spectral_normalize_torch(torch.matmul(mel_basis, stft_spec))

        return mel[0], stft_spec[0]

    def pad_spec(self, log_mel_spec):
        n_frames = log_mel_spec.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[0 : self.target_length, :]

        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]

        return log_mel_spec

    def wav_feature_extraction(self, waveform):  # (1, 163840)
        waveform = waveform[0, ...]  # (163840,)
        waveform = torch.FloatTensor(waveform).to(self.device)
        log_mel_spec, stft = self.mel_spectrogram_train(waveform.unsqueeze(0))  # input: torch.Size([1, 163840])

        if isinstance(log_mel_spec, np.ndarray):
            log_mel_spec = torch.FloatTensor(log_mel_spec.T)
            stft = torch.FloatTensor(stft.T)  #!! 여기서부터는 안되어있다고 보면 됨.
        elif isinstance(log_mel_spec, torch.Tensor):
            log_mel_spec = log_mel_spec.T.float()
            stft = stft.T.float()
        else:
            raise TypeError("numpy array, torch.Tensor are only supported.")

        log_mel_spec, stft = self.pad_spec(log_mel_spec), self.pad_spec(stft)  #!!
        return log_mel_spec, stft

    def read_audio_file(self, filename):
        # 1. 오디오 파일 로드 또는 빈 파형 생성
        if os.path.exists(filename):
            waveform, random_start = self.read_wav_file(filename)
        else:
            target_length = int(self.sampling_rate * self.duration)
            waveform = torch.zeros((1, target_length))
            random_start = 0
            print(f'Non-fatal Warning [dataset.py]: The wav path "{filename}" not found. Using empty waveform.')
        # 2. 특성 추출 (필요한 경우)
        log_mel_spec, stft = (None, None) if self.waveform_only else self.wav_feature_extraction(waveform)
        return log_mel_spec, stft, waveform, random_start
        
    def preprocessing_data(self, file_path):
        log_mel_spec, stft, waveform, random_start = self.read_audio_file(file_path)
        waveform = torch.FloatTensor(waveform)

        filename = os.path.splitext(os.path.basename(file_path))[0]
        text = filename.replace('_', ' ')
        print(text)
        if text is None:
            print("Warning: The model return None on key text", filename); text = ""


        assert waveform.shape
        data = {
            "text": [text],  # list
            "fname": [filename],  # list
            "waveform": "" if (waveform is None) else waveform.float(),  # tensor, [B, 1, samples_num]
            "stft": "" if (stft is None) else stft.float(),  # tensor, [B, t-steps, f-bins]          #!! 된거임
            "log_mel_spec": "" if (log_mel_spec is None) else log_mel_spec.float(),  # tensor, [B, t-steps, mel-bins]
            }  # return 할때 기준 shape
        
        for key, value in data.items():
            if key in ["waveform", "stft", "log_mel_spec"]:
                data[key] = value.unsqueeze(0)    #!! 된거임

        assert data["waveform"].shape == torch.Size([1, 1, 163840]), data["waveform"].shape
        assert data["stft"].shape == torch.Size([1, 1024, 512]), data["stft"].shape
        assert data["log_mel_spec"].shape == torch.Size([1, 1024, 64]), data["log_mel_spec"].shape
        
        return data

    def get_mixed_batches(self, set1, set2, snr_db=0):
        wav1, wav2 = set1["waveform"], set2["waveform"]
        assert wav1.shape == wav2.shape
        power1, power2 = torch.mean(wav1 ** 2), torch.mean(wav2 ** 2)
        scaling_factor = torch.sqrt(power1 / power2 * 10 ** (-snr_db/10))
        mixed = wav1 + wav2 * scaling_factor
        max_abs = mixed.abs().max()
        if max_abs > 1:
            mixed /= max_abs
        mixed_wav = (mixed * 0.5).float()
        log_mel_spec, stft = self.mel_spectrogram_train(mixed_wav[0, ...])  # input: torch.Size([1, 163840])
        if isinstance(log_mel_spec, np.ndarray):
            log_mel_spec = self.pad_spec(torch.FloatTensor(log_mel_spec.T)).unsqueeze(0).float()
        elif isinstance(log_mel_spec, torch.Tensor):
            log_mel_spec = self.pad_spec(log_mel_spec.T.float()).unsqueeze(0).float()
        else:
            raise TypeError("numpy array, torch.Tensor are only supported.")
        mixed_wav, log_mel_spec, stft = mixed_wav.to(self.device), log_mel_spec.to(self.device), stft.unsqueeze(0).float().to(self.device)  #####
        def set_dict(batch):
            return {
                "text": batch["text"],        # List
                "fname": batch["fname"],      # List
                "waveform": mixed_wav,        # Tensor, [B, 1, samples_num]
                "stft": stft,                 # Tensor, [B, f-bins, t-steps]  <-- 여기는 이거 아님
                "log_mel_spec": log_mel_spec, # Tensor, [B, t-steps, mel-bins]
            }
        mixed_set1 = set_dict(set1)
        mixed_set2 = set_dict(set2)
        assert mixed_set1["waveform"].shape == torch.Size([1, 1, 163840]), mixed_set1["waveform"].shape
        assert mixed_set1["stft"].shape == torch.Size([1, 513, 1024]), mixed_set1["stft"].shape
        assert mixed_set1["log_mel_spec"].shape == torch.Size([1, 1024, 64]), mixed_set1["log_mel_spec"].shape

        return (mixed_set1, mixed_set2)

    ######### 추가된 부분
    def wav_to_fbank(self, filename, target_length=1024, fn_STFT=None):
        assert fn_STFT is not None

        # mixup
        waveform, _ = self.read_wav_file(filename)  # hop size is 160
        waveform = waveform / np.max(np.abs(waveform))
        waveform = 0.5 * waveform

        waveform = waveform[0, ...]
        waveform = torch.FloatTensor(waveform)

        fbank, log_magnitudes_stft, energy = self.get_mel_from_wav(waveform, fn_STFT)

        fbank = torch.FloatTensor(fbank.T)
        log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

        fbank, log_magnitudes_stft = self.pad_spec(fbank), self.pad_spec(log_magnitudes_stft)

        return fbank, log_magnitudes_stft, waveform

    def get_mel_from_wav(self, wav, fn_stft):
        wav = torch.clip(torch.FloatTensor(wav).unsqueeze(0), -1, 1)
        wav = torch.autograd.Variable(wav, requires_grad=False)
        package = fn_stft.mel_spectrogram(wav)
        print(package)
        melspec, log_magnitudes_stft, energy, _ = package
        for i in package:
            print(i.shape)
        melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
        log_magnitudes_stft = (torch.squeeze(log_magnitudes_stft, 0).numpy().astype(np.float32))
        energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
        return melspec, log_magnitudes_stft, energy
    ####### 여기까지

    