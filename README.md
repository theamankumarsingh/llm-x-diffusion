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
| `SYSTEM_PROMPT_PATH` | `src/llm_x_diffusion/system_prompt/default.md` | Path to the LLM system instructions |
| `SEED` | `42` | For reproducibility |
| `STEPS` | `50` | Denoising iterations |
| `GUIDANCE_SCALE` | `7.5` | Prompt adherence (CFG) |
| `TEMPERATURE` | `0.7` | LLM creativity |
| `NEGATIVE_PROMPT` | `""` | Default terms to avoid |

**Customizing Prompts**: If `SYSTEM_PROMPT_PATH` is not specified, the tool uses the default system prompt. To use your own instructions, specify the path to your `.md` file in the `.env` configuration.

## Output
Results are saved in `artifacts/<timestamp>/`:
- `output.png`: The generated image.
- `info.json`: Comprehensive run metadata, including the model versions, system and expanded prompts, device used, and all hyperparameters.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
