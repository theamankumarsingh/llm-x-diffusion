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

from dotenv import load_dotenv
load_dotenv()

import os, sys, torch
from datetime import datetime
from .diffusion.diffusers import generate, load_diffusion, encode_text
from .llm.transformers import generate_text, load_llm
from .utils.io import save_results

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def main():
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.use_deterministic_algorithms(True)
    
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a photo of a cat riding a skateboard"
    neg_prompt = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("NEGATIVE_PROMPT", "")
    
    mode = "pipeline"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx+1]

    # Configurable Parameters from ENV
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env is not None else torch.randint(0, 2**32, (1,)).item()
    steps, guidance_scale, temperature = int(os.environ.get("STEPS", 50)), float(os.environ.get("GUIDANCE_SCALE", 7.5)), float(os.environ.get("TEMPERATURE", 0.7))
    
    torch.manual_seed(seed)
    unet, vae, sched, d_tok, t_enc = load_diffusion(os.environ.get("DIFFUSION_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5"), dev)
    
    max_tokens = d_tok.model_max_length
    if mode == "pipeline":
        tok, llm = load_llm(os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"), dev)
        llm_out = generate_text(tok, llm, [prompt], dev, max_tokens=max_tokens, temperature=temperature)[0]
    else:
        llm_out = prompt

    pos_emb, neg_emb = encode_text(d_tok, t_enc, llm_out, dev), encode_text(d_tok, t_enc, neg_prompt, dev)
    img, _, _ = generate(unet, vae, sched, pos_emb, neg_emb, seed, steps=steps, guidance_scale=guidance_scale)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    info = {
        "timestamp": timestamp, "input_prompt": prompt, "llm_output": llm_out, "negative_prompt": neg_prompt,
        "steps": steps, "guidance_scale": guidance_scale, "temperature": temperature, "max_tokens": max_tokens,
        "seed": seed, "diffusion_input": pos_emb.tolist(), "diffusion_output": f"Image saved to artifacts/{timestamp}/output.png",
        "mode": mode
    }
    save_results(img, info, timestamp)

if __name__ == "__main__":
    main()
