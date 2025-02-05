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
        self.min_step = int(self.num_train_timesteps * 0.20)
        self.max_step = int(self.num_train_timesteps * 0.90)
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
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.betas = self.scheduler.betas.to(self.device)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), self.alphas_cumprod[:-1]]).to(self.device)

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
        x = x.half()  ##
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
        latents = latents.half()  ##
        mel_spectrogram = self.vae.decode(latents).sample
        return mel_spectrogram
    
    def mel_spectrogram_to_waveform(self, mel_spectrogram):  # don't use for final postprocessing
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)

        waveform = self.vocoder(mel_spectrogram)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        waveform = waveform.cpu().float()
        return waveform

    ########### 여기부터 추가됨 Editing
    def style_transfer(
        self,
        text,  #
        original_audio_file_path,  #
        transfer_strength,  #
        seed=42,
        duration=10,
        batchsize=1,
        guidance_scale=2.5,
        ddim_steps=199,
        processor=None,
    ):
        device = self.device
        assert original_audio_file_path is not None, "You need to provide the original audio file path"
        audio_file_duration = get_duration(original_audio_file_path)
        assert get_bit_depth(original_audio_file_path) == 16, "The bit depth of the original audio file %s must be 16" % original_audio_file_path
        if(duration > audio_file_duration):
            print("Warning: Duration you specified %s-seconds must equal or smaller than the audio file duration %ss" % (duration, audio_file_duration))
            duration = round_up_duration(audio_file_duration)
            print("Set new duration as %s-seconds" % duration)

        seed_everything(int(seed))

        fn_STFT = processor.STFT

        mel, _, _ = processor.wav_to_fbank(original_audio_file_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT)
        mel = mel.unsqueeze(0).unsqueeze(0).to(device)
        from einops import repeat
        mel = repeat(mel, "1 ... -> b ...", b=batchsize)
        init_latent_x = self.encode_audios(mel)  # move to latent space, encode and sample
        if(torch.max(torch.abs(init_latent_x)) > 1e2):
            init_latent_x = torch.clip(init_latent_x, min=-10, max=10)
        sampler = DDIMSampler(self)  ### latent_diffusion
        sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=1.0, verbose=False)  ##

        t_enc = int(transfer_strength * ddim_steps)
        prompts = text

        with torch.no_grad():
            # with self.ema_scope():

            # uc = None
            # if guidance_scale != 1.0:
            #     uc = latent_diffusion.cond_stage_model.get_unconditional_condition(batchsize)
            # c = latent_diffusion.get_learned_conditioning([prompts] * batchsize)
            uncond, cond = self._encode_prompt(prompts=prompts).chunk(2)
            print(uncond.shape, cond.shape)
            z_enc = sampler.stochastic_encode(init_latent_x, torch.tensor([t_enc] * batchsize).to(device))  ##
            samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=guidance_scale, unconditional_conditioning=uncond,)  ##
            x_samples = self.decode_latents(samples)  # decode_first_stage
            x_samples = self.decode_latents(samples[:,:,:-3,:])
            print(x_samples.shape)
            waveform = self.mel_spectrogram_to_waveform(x_samples)
        
            # if audio.dim() == 2:
            #     audio = audio[:, :self.original_waveform_length]
            # else:
            #     audio = audio[:self.original_waveform_length]
            
            # if output_type == "np":
            #     audio = audio.detach().numpy()
            
            # if not return_dict:
            #     return (audio,)
            # return audio
            #     waveform = self.post_process_from_mel(x_samples)

        return waveform


    def super_resolution_and_inpainting(
        self,
        latent_diffusion,  ##
        text,
        original_audio_file_path = None,
        seed=42,
        ddim_steps=200,
        duration=None,
        batchsize=1,
        guidance_scale=2.5,
        n_candidate_gen_per_text=3,
        time_mask_ratio_start_and_end=(0.10, 0.15), # regenerate the 10% to 15% of the time steps in the spectrogram
        # time_mask_ratio_start_and_end=(1.0, 1.0), # no inpainting
        # freq_mask_ratio_start_and_end=(0.75, 1.0), # regenerate the higher 75% to 100% mel bins
        freq_mask_ratio_start_and_end=(1.0, 1.0), # no super-resolution
        processor=None,
    ):
        seed_everything(int(seed))

        fn_STFT = processor.STFT
        
        # waveform = read_wav_file(original_audio_file_path, None)
        mel, _, _ = processor.wav_to_fbank(original_audio_file_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT)
        
        batch = make_batch_for_text_to_audio(text, fbank=mel[None,...], batchsize=batchsize)

        # latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
        # latent_diffusion = set_cond_text(latent_diffusion)
            
        with torch.no_grad():
            waveform = latent_diffusion.generate_sample_masked(  ##
                [batch],
                unconditional_guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
                n_candidate_gen_per_text=n_candidate_gen_per_text,
                duration=duration,
                time_mask_ratio_start_and_end=time_mask_ratio_start_and_end,
                freq_mask_ratio_start_and_end=freq_mask_ratio_start_and_end
            )
        return waveform
    


import torch
import numpy as np
from tqdm import tqdm

from src_audioldm.utilities.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    extract_into_tensor,
)

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.modelpipe = model.pipe
        self.ddpm_num_timesteps = model.num_train_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert (
            alphas_cumprod.shape[0] == self.ddpm_num_timesteps
        ), "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),)
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),)

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer("ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}"
                    )
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                    )

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        samples, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
        )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        device = self.modelpipe.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = (
                self.ddpm_num_timesteps
                if ddim_use_original_steps
                else self.ddim_timesteps
            )
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = (
                int(
                    min(timesteps / self.ddim_timesteps.shape[0], 1)
                    * self.ddim_timesteps.shape[0]
                )
                - 1
            )
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {"x_inter": [img], "pred_x0": [img]}
        time_range = (
            reversed(range(0, timesteps))
            if ddim_use_original_steps
            else np.flip(timesteps)
        )
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        # iterator = gr.Progress().tqdm(time_range, desc="DDIM Sampler", total=total_steps)
        iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            if mask is not None:
                assert x0 is not None
                img_orig = self.modelpipe.q_sample(
                    x0, ts
                )  # TODO deterministic forward pass?
                img = (
                    img_orig * mask + (1.0 - mask) * img
                )  # In the first sampling step, img is pure gaussian noise

            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
            img, pred_x0 = outs
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates["x_inter"].append(img)
                intermediates["pred_x0"].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)

        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0
                + extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(
        self,
        x_latent,
        cond,
        t_start,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        use_original_steps=False,
    ):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc="Decoding image", total=total_steps)
        x_dec = x_latent

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            if step < 0:  #####
                print(f"Warning: step value is negative ({step}). Setting step = 0.")
                step = 0  # 최소 0 이상으로 설정  #####

            if x_latent.shape[0] == 0:
                raise ValueError("Error: x_latent has zero elements. Check the input to the decoder.")

            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(
                x_dec,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
            )
        return x_dec

    @torch.no_grad()
    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
            e_t = self.model.unet(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            x_in = x_in.to(torch.float16)
            t_in = t_in.to(torch.float16)
            c_in = c_in.to(torch.float16)
            e_t_uncond, e_t = self.model.unet(x_in, t_in, encoder_hidden_states=None, class_labels=c_in, cross_attention_kwargs=None).sample.chunk(2)
            # When unconditional_guidance_scale == 1: only e_t
            # When unconditional_guidance_scale == 0: only unconditional
            # When unconditional_guidance_scale > 1: add more unconditional guidance
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.modelpipe.parameterization == "eps"
            e_t = score_corrector.modify_score(
                self.modelpipe, e_t, x, t, c, **corrector_kwargs
            )

        alphas = self.modelpipe.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (
            self.modelpipe.alphas_cumprod_prev
            if use_original_steps
            else self.ddim_alphas_prev
        )
        sqrt_one_minus_alphas = (
            self.modelpipe.sqrt_one_minus_alphas_cumprod
            if use_original_steps
            else self.ddim_sqrt_one_minus_alphas
        )
        sigmas = (
            self.modelpipe.ddim_sigmas_for_original_num_steps
            if use_original_steps
            else self.ddim_sigmas
        )
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
        )

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.modelpipe.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise  # TODO
        return x_prev, pred_x0











def round_up_duration(duration):
    return int(round(duration/2.5) + 1) * 2.5

import wave
import contextlib

def get_duration(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)

def get_bit_depth(fname):
    with contextlib.closing(wave.open(fname, 'r')) as f:
        bit_depth = f.getsampwidth() * 8
        return bit_depth

def make_batch_for_text_to_audio(text, waveform=None, fbank=None, batchsize=1):
    text = [text] * batchsize
    if batchsize < 1:
        print("Warning: Batchsize must be at least 1. Batchsize is set to .")
    
    if(fbank is None):
        fbank = torch.zeros((batchsize, 1024, 64))  # Not used, here to keep the code format
    else:
        fbank = torch.FloatTensor(fbank)
        fbank = fbank.expand(batchsize, 1024, 64)
        assert fbank.size(0) == batchsize

    stft = torch.zeros((batchsize, 1024, 512))  # Not used

    if(waveform is None):
        waveform = torch.zeros((batchsize, 160000))  # Not used
    else:
        waveform = torch.FloatTensor(waveform)
        waveform = waveform.expand(batchsize, -1)
        assert waveform.size(0) == batchsize
        
    fname = [""] * batchsize  # Not used
    
    batch = (fbank, stft, None, fname, waveform, text)
    return batch

def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

