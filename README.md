# Cortex Deployer

Local control plane for open models. One process, a browser UI, and an OpenAI-compatible endpoint — the same idea as `npx @deepseek-ai/dsh web`, aimed at **managing inference backends** the way [Cortex Backends](https://cortex.shizuha.com/cortex/backends) does for the fleet.

Windows, Linux, and macOS.

```bash
pip install git+https://github.com/shizuha-labs/cortex-deployer.git
cortex-deployer server
# open http://127.0.0.1:7480
```

Or from a checkout: `python -m cortex_deployer server`.

## What you get

| Surface | Role |
|---|---|
| Web UI `http://127.0.0.1:7480` | List / start / stop / register backends |
| `GET/POST /api/backends` | Same data as the UI |
| `GET /v1/models` · `POST /v1/chat/completions` | Fan-out to whatever is healthy locally |
| `cortex-deployer connect` | Dial a Cortex router so the model shows up on the public catalog |

Engines: **llama.cpp**, **SGLang**, **vLLM**, **MLX**.

## Windows + RTX 5080 (Qwen3.8-27B)

1. Install Python 3.11+ and a CUDA `llama-server` on `PATH` (or set `CORTEX_DEPLOYER_LLAMA_SERVER` to the `.exe`).
2. Download a GGUF that fits **16 GB**. Q4 (~18 GB) usually needs a 24 GB card or CPU offload. Prefer **UD-Q3_K_XL** (or Q3) first; drop to Q2 if VRAM still OOMs.
3. `cortex-deployer server`
4. Open the UI → **Deploy model** → recipe `qwen38-27b-llamacpp.yaml` → set **Weights path** to the GGUF → **Start**.
5. When the row is healthy, `GET http://127.0.0.1:7480/v1/models` lists `Qwen3.8-27B-Q4` (or the name you typed).
6. Optional: **Register URL** if `llama-server` is already running.

State lives in `%USERPROFILE%\.cortex-deployer\` (override with `CORTEX_DEPLOYER_HOME`).

## CLI

```bash
cortex-deployer server --host 127.0.0.1 --port 7480   # also: up, web
cortex-deployer engines
cortex-deployer recipes
cortex-deployer render cortex_deployer/recipes/examples/llamacpp-cuda.yaml --json
cortex-deployer connect --gateway wss://…/deployer/ws/register --token … --model … --upstream http://127.0.0.1:8080/v1
```

`connect` never invents a token.

## Repository fence

Development: Origin `shizuha-labs/cortex-deployer-beta`.
Public: GitHub `shizuha-labs/cortex-deployer` via leak-checked merge commits only.

See `CONTRIBUTING.md` and `SECURITY.md`.
