from typing import Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
   AutoencoderKL,
   PNDMScheduler,
   AudioLDMPipeline,
   UNet2DConditionModel,
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
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.scale_factor = 1.0
                
        # Initialize pipeline with PNDM scheduler (load from pipeline. let us use dreambooth models.)
        pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        
        #### 둘중 택일
        pipe.scheduler=PNDMScheduler(beta_start=0.00085,beta_end=0.012,beta_schedule="scaled_linear",num_train_timesteps=self.num_train_timesteps)
        # pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler.config.skip_prk_steps = True  # PNDM의 초기 단계 스킵
        # pipe.scheduler.config.set_alpha_to_one = False  # DDIM과 비슷한 scaling 유지

        # Setup components and move to device
        self.pipe = pipe
        self.components = {
            'vae': (pipe.vae, AutoencoderKL),
            'tokenizer': (pipe.tokenizer, RobertaTokenizerFast),
            'text_encoder': (pipe.text_encoder, ClapTextModelWithProjection),
            'unet': (pipe.unet, UNet2DConditionModel),
            'vocoder': (pipe.vocoder, SpeechT5HifiGan),
            'scheduler': (pipe.scheduler, PNDMScheduler)
        }
        
        # Initialize and validate components
        for name, (component, expected_type) in self.components.items():
            if name in ['vae', 'text_encoder', 'unet', 'vocoder']:
                component = component.to(self.device)
            assert isinstance(component, expected_type), f"{name} type mismatch: {type(component)}"
            setattr(self, name, component)
        
        self.uncond_text = ''
        self.checkpoint_path = repo_id
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.audio_length_in_s = 10.24
        self.original_waveform_length = int(self.audio_length_in_s * self.vocoder.config.sampling_rate)
        print(f'[INFO] audioldm.py: loaded AudioLDM!')

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
        return prompt_embeds

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

    def get_input(self, batch, key):
        '''
        fname = batch["fname"]
        text = batch["text"]
        label_indices = batch["label_vector"]
        waveform = batch["waveform"]
        stft = batch["stft"]
        mel = batch["log_mel_spec"]
        '''
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
    
    def encode_audios(self, x):
        encoder_posterior = self.vae.encode(x)

        if hasattr(encoder_posterior.latent_dist, "sample"):  # sample() 메서드가 있는 경우 (DiagonalGaussianDistribution 타입)
            unscaled_z = encoder_posterior.latent_dist.sample()
        elif isinstance(encoder_posterior.latent_dist, torch.Tensor):  # 그냥 Tensor라면 그대로 사용
            unscaled_z = encoder_posterior.latent_dist
        else:
            raise NotImplementedError(f"Unsupported encoder_posterior type: {type(encoder_posterior)}")

        # self.scale_factor = 1.0 / z.flatten().std()  # Normalize z to have std=1  #### 둘 다 값 확인해보고 뭐 쓸지 결정
        self.scale_factor = self.vae.config.scaling_factor  # Normalize z to have std=1  ####
        # print(self.scale_factor)  #### 몇인지 확인하자 : 0.9227914214134216
        z = self.scale_factor * unscaled_z
        return z

    def post_process_from_latent(self, latents, output_type: Optional[str] = "np", return_dict: bool = True):
        mel_spectrogram = self.decode_latents(latents)
        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
        audio = audio[:, :self.original_waveform_length]
        
        if output_type == "np":
            audio = audio.detach().numpy()
        
        if not return_dict:
            return (audio,)
        return audio
    
    def post_process_from_mel(self, mel_spectrogram, output_type: Optional[str] = "np", return_dict: bool = True):
        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
        if audio.dim() == 2:
            audio = audio[:, :self.original_waveform_length]
        else:
            audio = audio[:self.original_waveform_length]
        
        if output_type == "np":
            audio = audio.detach().numpy()
        
        if not return_dict:
            return (audio,)
        return audio

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        mel_spectrogram = self.vae.decode(latents).sample
        return mel_spectrogram
    
    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform