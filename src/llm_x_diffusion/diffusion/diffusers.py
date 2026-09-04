# Copyright 2026 Aman Kumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from PIL import Image
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

def load_diffusion(name, device):
    return (UNet2DConditionModel.from_pretrained(name, subfolder="unet").to(device).eval().requires_grad_(False),
            AutoencoderKL.from_pretrained(name, subfolder="vae").to(device).eval().requires_grad_(False),
            DDIMScheduler.from_pretrained(name, subfolder="scheduler"),
            CLIPTokenizer.from_pretrained(name, subfolder="tokenizer"),
            CLIPTextModel.from_pretrained(name, subfolder="text_encoder").to(device).eval().requires_grad_(False))

@torch.no_grad()
def encode_text(tokenizer, text_encoder, text, device):
    return text_encoder(**tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)).last_hidden_state

@torch.no_grad()
def generate(unet, vae, scheduler, embedding, negative_embedding, seed, steps=50, guidance_scale=7.5):
    scheduler.set_timesteps(steps)
    gen = torch.Generator(device=embedding.device).manual_seed(seed) if seed else None
    latents = torch.randn((embedding.shape[0], unet.config["in_channels"], unet.config["sample_size"], unet.config["sample_size"]), device=embedding.device, dtype=unet.dtype, generator=gen)
    embedding = embedding.to(unet.dtype)
    negative_embedding = negative_embedding.to(unet.dtype)
    for t in scheduler.timesteps:
        noise_pred_uncond = unet(latents, t, encoder_hidden_states=negative_embedding).sample
        noise_pred_text = unet(latents, t, encoder_hidden_states=embedding).sample
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    img = vae.decode(latents / vae.config.scaling_factor).sample
    return Image.fromarray(((img / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()[0] * 255).round().astype("uint8")), latents, scheduler.timesteps.tolist()
