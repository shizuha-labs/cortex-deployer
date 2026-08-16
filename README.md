# Cortex Deployer

Local control plane for open models. One process, a browser UI, and an OpenAI-compatible endpoint — the same idea as `npx @deepseek-ai/dsh web`, aimed at **managing inference backends** the way [Cortex Backends](https://cortex.shizuha.com/cortex/backends) does for the fleet.

Windows, Linux, and macOS.

```bash
curl -fsSL https://cortex.shizuha.com/deployer/install.sh | bash
cortex-deployer server
# open http://127.0.0.1:7480  →  Choose a Qwen build
```

The installer does **not** use the OS `pip` (Debian/Ubuntu PEP 668). It bootstraps an isolated `uv` + CPython under `~/.cortex-deployer` and puts a wrapper on `~/.local/bin`.

Windows PowerShell:

```powershell
irm https://cortex.shizuha.com/deployer/install.ps1 | iex
cortex-deployer server
```

Or from a checkout: `python -m cortex_deployer server`.

**Choose a Qwen build** lists Q2 / Q3 / Q4 (and MLX on Apple) from the live catalog at [cortex.shizuha.com/deployer](https://cortex.shizuha.com/deployer), marks one as recommended from this GPU's VRAM, then installs official `llama-server` + that GGUF. `cortex-deployer setup --recipe …` is the headless path. No separate CUDA toolkit.

## What you get

| Surface | Role |
|---|---|
| Web UI `http://127.0.0.1:7480` | One-click setup, list / start / stop / register backends, GPU fit, download, chat |
| `POST /api/setup` | Recommended recipe + engine + weights + start |
| `GET/POST /api/backends` | Same data as the UI |
| `GET /api/recommend` · `/api/downloads` | GPU-aware recipe picker and Hugging Face pulls |
| `GET /v1/models` · `POST /v1/chat/completions` | Fan-out (CORS + stream) to whatever is healthy locally |
| `cortex-deployer connect` | Dial a Cortex router so the model shows up on the public catalog |

Engines: **llama.cpp**, **SGLang**, **vLLM**, **MLX**.

## Windows + RTX 5080 (Qwen3.8-27B)

16 GB cards use **Q3** (`UD-Q3_K_XL` ~13.4 GB). Q4 is marked tight/skip.

1. NVIDIA Game Ready / Studio driver (the only host install).
2. In PowerShell:

```powershell
irm https://cortex.shizuha.com/deployer/install.ps1 | iex
cortex-deployer server
```

3. Open http://127.0.0.1:7480 and click **Choose a Qwen build**.

The picker shows every Qwen3.8-27B quant from [cortex.shizuha.com/deployer](https://cortex.shizuha.com/deployer). On a 16 GB 5080, **Q3 is recommended**, Q2 is the fallback, Q4 is marked won't-fit. Picking one:

- downloads official `llama.cpp` **Windows CUDA 13** + matching CUDA DLLs
- pulls that Unsloth GGUF
- starts `llama-server` and lists the served name at `/v1/models`

Chat in the same page, or point any OpenAI client at `http://127.0.0.1:7480/v1`. Headless: `cortex-deployer setup`.

Optional **Cortex** on a row announces it to a Cortex gateway (token from the Cortex UI). `connect` never invents a token.

State lives in `%USERPROFILE%\.cortex-deployer\` (override with `CORTEX_DEPLOYER_HOME`).

## CLI

```bash
cortex-deployer server --host 127.0.0.1 --port 7480   # also: up, web
cortex-deployer setup                                 # one-click recommended Qwen
cortex-deployer recommend
cortex-deployer download --repo unsloth/Qwen3.8-27B-GGUF --glob '*UD-Q3_K_XL.gguf'
cortex-deployer engines
cortex-deployer recipes
cortex-deployer render cortex_deployer/recipes/examples/qwen38-27b-q3-llamacpp.yaml --json
cortex-deployer run cortex_deployer/recipes/examples/qwen38-27b-mlx.yaml   # foreground; launchd/systemd
cortex-deployer connect --gateway wss://…/deployer/ws/register --token … --model … --upstream http://127.0.0.1:8080/v1
```

## Repository fence

Development: Origin `shizuha-labs/cortex-deployer-beta`.
Public: GitHub `shizuha-labs/cortex-deployer` via leak-checked merge commits only.

See `CONTRIBUTING.md` and `SECURITY.md`.
