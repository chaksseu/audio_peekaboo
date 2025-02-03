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
   RobertaTokenizer,
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
                
        # Initialize pipeline with PNDM scheduler (load from pipeline. let us use dreambooth models.)
        pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        pipe.scheduler=PNDMScheduler(beta_start=0.00085,beta_end=0.012,beta_schedule="scaled_linear",num_train_timesteps=self.num_train_timesteps)
        
        # Setup components and move to device
        self.pipe = pipe
        self.components = {
            'vae': (pipe.vae, AutoencoderKL),
            'tokenizer': (pipe.tokenizer, RobertaTokenizer),
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
        print(f'[INFO] audioldm.py: loaded AudioLDM!')

    def get_text_embeddings(self, prompts: Union[str, List[str]]) -> torch.Tensor:
        prompts = [prompts] if isinstance(prompts, str) else prompts

        def get_embeddings(text_list):
            tokens = self.tokenizer(
                text_list,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt'
            ).input_ids
            
            with torch.no_grad():
                return self.text_encoder(tokens.to(self.device))[0]

        # Get text and unconditional embeddings
        text_embeddings = get_embeddings(prompts)  # [B, 77, 768]
        uncond_embeddings = get_embeddings([self.uncond_text] * len(prompts))  # [B, 77, 768]
        assert (uncond_embeddings == uncond_embeddings[0][None]).all()  # All the same
        
        return torch.cat([uncond_embeddings, text_embeddings])  # First B rows: uncond, Last B rows: cond

    def _encode_prompt(self, prompts: Union[str, List[str]], device, do_classifier_free_guidance=True) -> torch.Tensor:
        
        # 1. Batch size 결정
        if prompts is not None and isinstance(prompts, str):
            batch_size = 1
        elif prompts is not None and isinstance(prompts, list):
            batch_size = len(prompts)
        else:
            batch_size = prompt_embeds.shape[0]

        # 2. Prompt embedding 생성
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids, attention_mask = text_inputs.input_ids.to(device), text_inputs.attention_mask.to(device)

        # Truncation 경고
        untruncated_ids = self.tokenizer(prompts, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(f"The following part of your input was truncated because CLAP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}")

        # Text embedding 계산 및 정규화
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask.to(device)).text_embeds
        # additional L_2 normalization over each hidden-state
        prompt_embeds = F.normalize(prompt_embeds, dim=-1).to(dtype=self.text_encoder.dtype, device=device)  # [B, 77, 768] >> ??
        print(prompt_embeds.shape)

        # 3. get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_input = self.tokenizer(
                [""] * batch_size,
                padding="max_length",
                max_length=prompt_embeds.shape[1],
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids, attention_mask = uncond_input.input_ids.to(device), uncond_input.attention_mask.to(device)
            uncond_prompt_embeds = self.text_encoder(uncond_input_ids, attention_mask=attention_mask).text_embeds
            # additional L_2 normalization over each hidden-state
            uncond_prompt_embeds = F.normalize(uncond_prompt_embeds, dim=-1)  # [B, 77, 768] >> ??
            print(uncond_prompt_embeds.shape)

            assert (uncond_prompt_embeds == uncond_prompt_embeds[0][None]).all()  # All the same
            prompt_embeds = torch.cat([uncond_prompt_embeds, prompt_embeds])  # First B rows: uncond, Last B rows: cond
        return prompt_embeds

    def train_step(self, text_embeddings: torch.Tensor, pred_rgb: torch.Tensor, guidance_scale: float = 100, t: Optional[int] = None):
        # Prepare image and timestep
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        t = t if t is not None else torch.randint(self.min_step, self.max_step + 1, [1], device=self.device)
        assert 0 <= t < self.num_train_timesteps, f'invalid timestep t={t}'

        # Encode image to latents (with grad)
        latents = self.encode_imgs(pred_rgb_512)

        # Predict noise without grad
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t.cpu())
            noise_pred = self.unet(torch.cat([latents_noisy] * 2), t, encoder_hidden_states=text_embeddings)['sample']

        # Guidance . High value from paper
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Calculate and apply gradients
        w = (1 - self.alphas_cumprod[t])
        grad = w * (noise_pred - noise)
        latents.backward(gradient=grad, retain_graph=True)

        noise_mse = ((noise_pred - noise) ** 2).mean().item()
        uncond, cond = noise_pred_uncond.abs().mean().item(), noise_pred_text.abs().mean().item()
        c_minus_unc = (noise_pred_uncond - noise_pred_text).abs().mean().item()
        return (noise_mse, uncond, cond, c_minus_unc)

    def encode_imgs(self, imgs:torch.Tensor)->torch.Tensor:
        imgs = 2 * imgs - 1  # [-1, 1]
        posterior = self.vae.encode(imgs)
        latents = posterior.latent_dist.sample() * self.VAE_SCALING_FACTOR  # [B, 3, H, W]
        return latents

    def get_input_from_dict(self, batch, key):
        '''
        fname = batch["fname"]
        text    batch["text"]
        label_indices    batch["label_vector"]
        waveform    batch["waveform"]
        stft    batch["stft"]
        fbank    batch["log_mel_spec"]
        '''
        return_format = {
            "fname": batch["fname"],
            "text": list(batch["text"]),
            "waveform": batch["waveform"].to(memory_format=torch.contiguous_format).float(),
            "stft": batch["stft"].to(memory_format=torch.contiguous_format).float(),
            "fbank": batch["log_mel_spec"].unsqueeze(1).to(memory_format=torch.contiguous_format).float(),
        }
        for key in batch.keys():
            if key not in return_format.keys():
                return_format[key] = batch[key]
        return return_format[key]
    
    def get_input(
        self,
        batch,
        k,
        return_first_stage_encode=True,
        return_decoding_output=False,
        return_encoder_input=False,
        return_encoder_output=False,
        unconditional_prob_cfg=0.1,
    ):
        
        x = self.get_input_from_dict(batch, k).to(self.device)