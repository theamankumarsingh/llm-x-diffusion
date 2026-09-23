# llm-x-diffusion

A streamlined pipeline that uses an LLM to expand simple user inputs into detailed visual prompts for a Diffusion model.

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your HF_TOKEN and model paths
   ```

3. **Run**:
   ```bash
   uv run llm-x-diffusion "A cat riding a skateboard" "blurry, low quality"
   ```
   *(The second argument for the negative prompt is optional)*

## Run Modes

The tool supports two operating modes using the `--mode` flag:

| Mode | Command | Description |
| :--- | :--- | :--- |
| `pipeline` (Default) | `--mode pipeline` | **LLM $\to$ Diffusion**: Expands the input prompt via the LLM before generating the image. |
| `diffusion` | `--mode diffusion` | **Diffusion Only**: Uses the input prompt directly. Skips the LLM stage to save memory and time. |

### Example: Diffusion-Only Mode
```bash
uv run llm-x-diffusion "A cat riding a skateboard" "blurry, low quality" --mode diffusion
```

## Configuration

Adjust these in `.env` or leave as defaults:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | LLM for prompt expansion |
| `DIFFUSION_MODEL` | `stable-diffusion-v1-5/...` | Base image model |
| `SEED` | `42` | For reproducibility |
| `STEPS` | `50` | Denoising iterations |
| `GUIDANCE_SCALE` | `7.5` | Prompt adherence (CFG) |
| `TEMPERATURE` | `0.7` | LLM creativity |
| `NEGATIVE_PROMPT` | `""` | Default terms to avoid |

## Output
Results are saved in `artifacts/<timestamp>/`:
- `output.png`: The generated image.
- `info.json`: Complete metadata of the run (prompts, seed, hyperparameters).

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
