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
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm(name, device):
    return AutoTokenizer.from_pretrained(name), AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval().requires_grad_(False)

@torch.no_grad()
def generate_text(tokenizer, model, prompts, device, system_prompt, max_tokens=100, temperature=0.7):
    texts = [tokenizer.apply_chat_template([{"role": "system", "content": system_prompt}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature)
    return [tokenizer.decode(out[inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip() for out in outputs]
