from typing import Any, Callable, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
   AudioLDMPipeline,
   AutoencoderKL,
   UNet2DConditionModel,
   DDIMScheduler,
)
from transformers import (
   ClapTextModelWithProjection,
   RobertaTokenizerFast,
   SpeechT5HifiGan,
   logging,
)

# Suppress partial model loading warning
logging.set_verbosity_error()
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class AudioLDM(nn.Module):
    
    def __init__(self, device='cuda', repo_id="cvssp/audioldm"):
        super().__init__()
        self.device = torch.device(device)
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.20)
        self.max_step = int(self.num_train_timesteps * 0.90)
        self.scale_factor = 1.0
                
        # Initialize pipeline with PNDM scheduler (load from pipeline. let us use dreambooth models.)
        pipe = AudioLDMPipeline.from_pretrained(repo_id)  # torch_dtype=torch.float16
        
        '''
        #### 둘중 택일
        ## pipe.scheduler=PNDMScheduler(beta_start=0.00085,beta_end=0.012,beta_schedule="scaled_linear",num_train_timesteps=self.num_train_timesteps)
        # pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler.config.skip_prk_steps = True  # PNDM의 초기 단계 스킵
        # pipe.scheduler.config.set_alpha_to_one = False  # DDIM과 비슷한 scaling 유지
        '''

        # Setup components and move to device
        self.pipe = pipe
        self.components = {
            'vae': (pipe.vae, AutoencoderKL),
            'tokenizer': (pipe.tokenizer, RobertaTokenizerFast),
            'text_encoder': (pipe.text_encoder, ClapTextModelWithProjection),
            'unet': (pipe.unet, UNet2DConditionModel),
            'vocoder': (pipe.vocoder, SpeechT5HifiGan),
            'scheduler': (pipe.scheduler, DDIMScheduler)
            # 'scheduler': (pipe.scheduler, PNDMScheduler)
        }
        
        # Initialize and validate components
        for name, (component, expected_type) in self.components.items():
            if name in ['vae', 'text_encoder', 'unet', 'vocoder']:
                component = component.to(self.device)
            assert isinstance(component, expected_type), f"{name} type mismatch: {type(component)}"
            setattr(self, name, component)
        
        self.uncond_text = ''
        self.checkpoint_path = repo_id
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.betas = self.scheduler.betas.to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]]).to(self.device)

        self.audio_length_in_s = 10.24
        self.original_waveform_length = int(self.audio_length_in_s * self.vocoder.config.sampling_rate)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        print(f'[INFO] audioldm.py: loaded AudioLDM!')
    
    def get_input(self, batch, key):
        return_format = {
            "fname": batch["fname"],
            "text": batch["text"],
            "waveform": batch["waveform"].to(memory_format=torch.contiguous_format).float(),
            "stft": batch["stft"].to(memory_format=torch.contiguous_format).float(),
            "mel": batch["log_mel_spec"].unsqueeze(1).to(memory_format=torch.contiguous_format).float(),
        }
        for key_ in batch.keys():
            if key_ not in return_format.keys():
                return_format[key_] = batch[key_]
        return return_format[key]
    
    def train_step(self, batch: dict, guidance_scale: float = 100, t: Optional[int] = None):
        # Prepare image and timestep

        x = self.get_input(batch, 'mel').to(self.device)  # mean: -4.63, std: 2.74
        text = self.get_input(batch, 'text')
        prompt_embeds = self._encode_prompt(text, do_classifier_free_guidance=True)

        t = t if t is not None else torch.randint(self.min_step, self.max_step + 1, (x.shape[0],), device=self.device).long()
        assert 0 <= t < self.num_train_timesteps, f'invalid timestep t={t}'
        x = x.half()  # Convert to half precision
        # Encode image to latents (with grad)
        latent = self.encode_audios(x)

        # Predict noise without grad
        with torch.no_grad():
            noise = torch.randn_like(latent)
            latents_noisy = self.scheduler.add_noise(latent, noise, t.cpu())
            noise_pred = self.unet(torch.cat([latents_noisy] * 2), t, encoder_hidden_states=None, class_labels=prompt_embeds, cross_attention_kwargs=None).sample

        # Guidance . High value from paper
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Calculate and apply gradients
        w = (1 - self.alphas_cumprod[t])
        grad = w * (noise_pred - noise)
        latent.backward(gradient=grad, retain_graph=True)

        noise_mse = ((noise_pred - noise) ** 2).mean().item()
        uncond, cond = noise_pred_uncond.abs().mean().item(), noise_pred_text.abs().mean().item()
        c_minus_unc = (noise_pred_uncond - noise_pred_text).abs().mean().item()
        return (noise_mse, uncond, cond, c_minus_unc)
    
    def _encode_prompt(self, prompts: Union[str, List[str]], do_classifier_free_guidance=True) -> torch.Tensor:
        # 1. Batch size 결정
        if prompts is not None and isinstance(prompts, str):
            batch_size = 1
        elif prompts is not None and isinstance(prompts, list):
            batch_size = len(prompts)
        else:
            raise ValueError(f"Invalid prompts: {prompts}")
        
        # 2. Prompt embedding 생성
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids, attention_mask = text_inputs.input_ids.to(self.device), text_inputs.attention_mask.to(self.device)

        # Truncation 경고
        untruncated_ids = self.tokenizer(prompts, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(f"The following part of your input was truncated because CLAP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}")

        # Text embedding 계산 및 정규화
        prompt_embeds = self.text_encoder(text_input_ids.to(self.device), attention_mask=attention_mask.to(self.device)).text_embeds
        # additional L_2 normalization over each hidden-state
        prompt_embeds = F.normalize(prompt_embeds, dim=-1).to(dtype=self.text_encoder.dtype, device=self.device)  # [B, 77, 768] >> [1,512]

        # 3. get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids, attention_mask = uncond_input.input_ids.to(self.device), uncond_input.attention_mask.to(self.device)
            uncond_prompt_embeds = self.text_encoder(uncond_input_ids, attention_mask=attention_mask).text_embeds
            # additional L_2 normalization over each hidden-state
            uncond_prompt_embeds = F.normalize(uncond_prompt_embeds, dim=-1)  # [B, 77, 768] >> ?? [1,512]

            assert (uncond_prompt_embeds == uncond_prompt_embeds[0][None]).all()  # All the same
            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])  # First B rows: uncond, Last B rows: cond
        return prompt_embeds\

    def encode_audios(self, x):
        encoder_posterior = self.vae.encode(x)
        unscaled_z = encoder_posterior.latent_dist.sample()
        # self.scale_factor = 1.0 / z.flatten().std()  # Normalize z to have std=1  #### 둘 다 값 확인해보고 뭐 쓸지 결정
        self.scale_factor = self.vae.config.scaling_factor  # Normalize z to have std=1  #### : 0.9227914214134216
        z = self.scale_factor * unscaled_z
        return z

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample
        return mel_spectrogram

    def mel_to_waveform(self, mel_spectrogram):  # don't use for final postprocessing
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # assert waveform.shape[1]==self.original_waveform_length, f'{waveform.shape}'
        print(waveform.shape)
        if waveform.dim() == 2:
            waveform = waveform[:, :self.original_waveform_length]
        elif waveform.dim() == 1:
            waveform = waveform[:self.original_waveform_length]
        else:
            raise ValueError
        waveform = waveform.detach().squeeze(0).cpu().float().numpy()

        return waveform  # [samples,]

    @torch.no_grad()
    def ddim_noising(
        self,
        latents: torch.Tensor,
        num_inference_steps: int = 50,
        transfer_strength: int = 1,
    ) -> torch.Tensor:
        r"""
        Args:
            latents (`torch.Tensor`): 초기 latents.
            num_inference_steps (`int`, default=50): noising step 수.
            transfer_strength (`int`, default=1): 변환 강도.
        Returns:
            `torch.Tensor`: 노이즈 추가된 latents.
        """
        device = latents.device
        # DDIM 전용 Scheduler라고 가정

        self.scheduler.set_timesteps(num_inference_steps, device=device)  
        all_timesteps = self.scheduler.timesteps  # 길이 50의 텐서 ex) [980, 960, ..., 0]

        t_enc = int(transfer_strength * num_inference_steps)
        used_timesteps = all_timesteps[-t_enc:]
        print(used_timesteps)
        noisy_latents = latents.clone()

        # forward로 t=0 -> t=1 ... -> t=T 방향으로 노이즈 주입
        for i, t in enumerate(reversed(used_timesteps)):
            noise = torch.randn_like(noisy_latents)
            # add_noise는 DDIMScheduler나 비슷한 스케줄러에 구현됨
            noisy_latents = self.scheduler.add_noise(noisy_latents, noise, t)

        return noisy_latents

    @torch.no_grad()
    def ddim_denoising(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        num_inference_steps: int = 50,
        transfer_strength: int = 1,
        guidance_scale: float = 7.5,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ) -> torch.Tensor:
        r"""
        DDIM Inversion 후 latents를 Denoising하는 메서드.

        Args:
            latents (`torch.Tensor`): 노이즈가 포함된 latents.
            prompt_embeds (`torch.Tensor`): text condition embedding.
            num_inference_steps (`int`, default=50): denoising step 수.
            transfer_strength (`int`, default=1): 변환 강도.
            guidance_scale (`float`, default=7.5): guidance scale.
            cross_attention_kwargs (`dict`, optional): cross attention 설정.
            callback (`Callable`, optional): 특정 step마다 호출할 함수.
            callback_steps (`int`, default=1): callback 호출 주기.

        Returns:
            `torch.Tensor`: Denoising된 latents.
        """

        device = latents.device
        do_classifier_free_guidance = guidance_scale > 1.0

        self.scheduler.set_timesteps(num_inference_steps, device=device)  
        all_timesteps = self.scheduler.timesteps

        t_enc = int(transfer_strength * num_inference_steps)
        used_timesteps = all_timesteps[-t_enc:]
        print(used_timesteps)
        
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)  # DDIM eta 설정

        num_warmup_steps = len(used_timesteps) - t_enc * self.scheduler.order
        print(num_warmup_steps)

        for i, t in enumerate(used_timesteps):
            # expand latents if classifier free guidance
            latent_model_input = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict noise
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=None,
                class_labels=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample

            # guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # DDIMScheduler의 step
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # callback
            if i == len(used_timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

        return latents


    def edit_audio_with_ddim(
        self,
        mel: torch.Tensor,
        text: Union[str, List[str]],
        duration: float,
        batch_size: int,
        transfer_strength: float,
        guidance_scale: float,
        ddim_steps: int,
        return_type: str = "pt",  # "pt" or "np"
        clipping = False,
    ):
        """
        오디오 파일에서 mel spectrogram 추출 후,
        DDIM Inversion을 통한 editing 수행 뒤 다시 waveform으로 복원하는 예시 메서드.

        Args:
            original_audio_file_path (str): 
                편집하고자 하는 원본 오디오 파일 경로.
            text (str or list[str]):
                generation or editing에 사용할 prompt.
            duration (float):
                편집에 사용할 오디오 길이(초). 원본 오디오 길이를 넘어선다면 자동 조정.
            batch_size (int):
                처리할 batch size.
            seed (int):
                재현성을 위한 seed 값.
            transfer_strength (float):
                0~1 범위 값으로, 몇 %의 ddim_steps까지 노이즈를 주입(Inversion)할지 결정.
            guidance_scale (float):
                Classifier-free guidance scale 값.
            ddim_steps (int):
                DDIM Inversion + Denoising에 사용할 step 수.
            return_type (str):
                "pt"면 `torch.Tensor` 형태, "np"면 `numpy.ndarray` 형태로 waveform을 반환.

        Returns:
            torch.Tensor 또는 numpy.ndarray:
                DDIM editing 이후의 waveform.
        """
        import torch
        import numpy as np
        from einops import repeat
        
        # ========== 사전 세팅 ==========
        # device = self._execution_device  # 또는 self.device / torch.device("cuda") 등 상황에 맞게
        # assert original_audio_file_path is not None, "original_audio_file_path를 지정해야 함"

        # 오디오 파일의 메타정보(샘플링 길이, bit depth 등) 가져오기
        audio_file_duration = 10.24
        # assert get_bit_depth(original_audio_file_path) == 16, (
        #     f"원본 오디오 {original_audio_file_path}의 bit depth는 16이어야 함"
        # )
        if duration > audio_file_duration:
            print(f"Warning: 지정한 duration {duration}s가 원본 오디오 길이 {audio_file_duration}s보다 큼")
            duration = 10.24 # round_up_duration(audio_file_duration)
            print(f"duration을 {duration}s로 조정")

        # 재현성을 위한 seed 설정
        # seed_everything(int(seed))

        # ========== 오디오 -> mel 변환 ==========

        # shape: (time, mel_bins)를 (1, 1, time, mel_bins)로 변환
        mel = mel.unsqueeze(0).unsqueeze(0).to(self.device)
        # batch_size만큼 반복
        mel = repeat(mel, "1 ... -> b ...", b=batch_size)

        # ========== mel -> latents ==========
        # 이 부분은 AudioLDM 파이프라인에서 제공하는 메서드라 가정(예: self.encode_audios 등)
        init_latent_x = self.encode_audios(mel)

        # clip 처리(너무 큰 값 방지)
        if torch.max(torch.abs(init_latent_x)) > 1e2:
            init_latent_x = torch.clamp(init_latent_x, min=-10.0, max=10.0)

        # ========== DDIM Inversion (noising) ==========
        # transfer_strength를 기반으로 t_enc 계산
        # t_enc = int(transfer_strength * ddim_steps)
        # prompt encoding
        # classifier-free guidance를 위해 uncond, cond 분리
        # 예시로, _encode_prompt에서 return을 torch.cat([uncond, cond]) 형태로 했다면 .chunk(2) 등으로 분리
        prompt_embeds = self._encode_prompt(prompts=text,
                                            do_classifier_free_guidance=True)
        uncond_embeds, cond_embeds = prompt_embeds.chunk(2)

        # t_enc 스텝까지 forward noising
        # ddim_noising 메서드를 활용한다
        # (필요에 따라 noising 전체 스텝은 ddim_steps로 진행하고, 
        #  실제 역으로 돌아갈 구간만 t_enc까지 사용할 수 있음)
        noisy_latents = self.ddim_noising(
            latents=init_latent_x,
            num_inference_steps=ddim_steps,  # 전체 step
            transfer_strength=transfer_strength,
        )
        # 위 예시는 전체 ddim_steps로 noise를 넣었지만, t_enc까지만 넣고 싶다면 적절히 조정
        
        # t_enc까지만 쓰고자 하는 경우, 
        # ex) noisy_latents = self.ddim_noising(..., num_inference_steps=t_enc)

        # ========== DDIM Denoising (editing) ==========
        # noised latents를 cond_embeds로 guidance 하여 다시 복원
        # (init_latent_x에 대한 구조적 활용을 원한다면, 메서드에 latent_x 등을 넘겨서 활용 가능)
        edited_latents = self.ddim_denoising(
            latents=noisy_latents,
            prompt_embeds=torch.cat([uncond_embeds, cond_embeds]),
            num_inference_steps=ddim_steps,
            transfer_strength=transfer_strength,
            guidance_scale=guidance_scale,
        )

        # ========== latent -> waveform ==========
        # mel spectrogram 복원
        mel_spectrogram = self.decode_latents(edited_latents)
        
        if clipping:
            mel_spectrogram = torch.maximum(torch.minimum(mel_spectrogram, mel), mel)

        # waveform 변환
        edited_waveform = self.mel_to_waveform(mel_spectrogram)

        # duration보다 긴 경우 자르기
        expected_length = int(duration * self.vocoder.config.sampling_rate)
        if edited_waveform.ndim == 2:
            edited_waveform = edited_waveform[:, :expected_length]
        else:
            edited_waveform = edited_waveform[:expected_length]

        # 반환 타입 결정
        if return_type == "np":
            edited_waveform = edited_waveform.cpu().numpy()
        # "pt"인 경우에는 torch.Tensor 그대로 반환

        return edited_waveform


if __name__ == '__main__':
    audioldm = AudioLDM(device='cpu')