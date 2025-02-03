from diffusers import AudioLDMPipeline

# hf_pipe = AudioLDMPipeline.from_pretrained(pretrained_model_name_or_path="cvssp/audioldm-s-full")

# print("UNet config:", hf_pipe.unet.config)
# print("\nText Encoder config:", hf_pipe.text_encoder.config)
# print("\nVAE config:", hf_pipe.vae.config)

import torch

repo_id = "cvssp/audioldm"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]
print(audio.shape)
