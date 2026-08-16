# Cortex Deployer

Local control plane for open models. One process, a browser UI, and an OpenAI-compatible endpoint — the same idea as `npx @deepseek-ai/dsh web`, aimed at **managing inference backends** the way [Cortex Backends](https://cortex.shizuha.com/cortex/backends) does for the fleet.

Windows, Linux, and macOS.

```bash
pip install git+https://github.com/shizuha-labs/cortex-deployer.git
cortex-deployer server
# open http://127.0.0.1:7480  →  Set up recommended Qwen
```

Or from a checkout: `python -m cortex_deployer server`.

**Set up recommended Qwen** (or `cortex-deployer setup`) does the rest: GPU fit, official `llama-server` install, Unsloth GGUF download, start. No separate CUDA toolkit or hand-placed binary.

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
2. Python 3.11+ from python.org — tick **Add python.exe to PATH**.
3. In PowerShell:

```powershell
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install "git+https://github.com/shizuha-labs/cortex-deployer.git"
cortex-deployer server
```

4. Open http://127.0.0.1:7480 and click **Set up recommended Qwen**.

That click:

- reads VRAM and picks `qwen38-27b-q3-llamacpp.yaml` (Q2 if you later OOM)
- downloads the latest official `llama.cpp` **Windows CUDA 13** zip plus matching CUDA runtime DLLs into `%USERPROFILE%\.cortex-deployer\engines\`
- pulls `unsloth/Qwen3.8-27B-GGUF` `*UD-Q3_K_XL*`
- starts `llama-server` and lists `Qwen3.8-27B-Q3` at `/v1/models`

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
