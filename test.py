from diffusers import AudioLDMPipeline
import os
# hf_pipe = AudioLDMPipeline.from_pretrained(pretrained_model_name_or_path="cvssp/audioldm-s-full")

# print("UNet config:", hf_pipe.unet.config)
# print("\nText Encoder config:", hf_pipe.text_encoder.config)
# print("\nVAE config:", hf_pipe.vae.config)

import torch

repo_id = "cvssp/audioldm"  # audioldm-s-full
torch.set_float32_matmul_precision("high")
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompts = [
    # 환경음 (Ambient & Nature Sounds)
    "Waves crashing on a beach",
    "Thunder rumbling in the distance",
    "A rainforest with birds chirping and leaves rustling",
    "A gentle breeze blowing through trees",
    "A campfire crackling at night",

    # 효과음 (Sound Effects / SFX)
    "A cat meowing",
    "A baby crying",
    "Footsteps on a wooden floor",
    "A door creaking open slowly",
    "A spaceship engine humming",
    "A robot beeping and glitching",

    # 단순한 음악 (Simple Music Loops)
    "A slow piano melody with soft chords",
    "A fast-paced techno beat with deep bass",
    "An 8-bit chiptune melody with simple drums",
    "A relaxing lo-fi hip-hop beat with vinyl crackle",
    "A synthwave melody with retro bass and dreamy pads"
]

audios = pipe(prompts, num_inference_steps=999, audio_length_in_s=10.24, guidance_scale=7.5).audios

import soundfile as sf
import re
os.makedirs("./samples", exist_ok=True)
# audio의 길이가 5.0초이고, sample rate는 16000임
for i, audio in enumerate(audios):
    caption = prompts[i]
    safe_filename = re.sub(r"[^\w\-_]", "_", caption)
    sf.write(f"./samples/{safe_filename}.wav", audio, 16000)