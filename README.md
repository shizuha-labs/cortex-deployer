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
| Web UI `http://127.0.0.1:7480` | List / start / stop / register backends, GPU fit, download, chat |
| `GET/POST /api/backends` | Same data as the UI |
| `GET /api/recommend` · `/api/downloads` | GPU-aware recipe picker and Hugging Face pulls |
| `GET /v1/models` · `POST /v1/chat/completions` | Fan-out (CORS + stream) to whatever is healthy locally |
| `cortex-deployer connect` | Dial a Cortex router so the model shows up on the public catalog |

Engines: **llama.cpp**, **SGLang**, **vLLM**, **MLX**.

## Windows + RTX 5080 (Qwen3.8-27B)

1. Install Python 3.11+ and a CUDA `llama-server.exe` on `PATH` (or `%USERPROFILE%\\llama.cpp\\`, or `CORTEX_DEPLOYER_LLAMA_SERVER`).
2. `cortex-deployer server` — the UI **recommends `qwen38-27b-q3-llamacpp.yaml`** on 16 GB cards. Q4 is marked tight/skip.
3. **Download recipe weights** (Hugging Face `unsloth/Qwen3.8-27B-GGUF` `*UD-Q3_K_XL*`) or point **Weights path** at a local GGUF.
4. Start. When healthy, `GET http://127.0.0.1:7480/v1/models` and the Chat box both work.
5. **Cortex** on a row announces it to a Cortex gateway (token from the Cortex UI).
6. If Q3 still OOMs, use `qwen38-27b-q2-llamacpp.yaml`.

State lives in `%USERPROFILE%\.cortex-deployer\` (override with `CORTEX_DEPLOYER_HOME`).

## CLI

```bash
cortex-deployer server --host 127.0.0.1 --port 7480   # also: up, web
cortex-deployer recommend
cortex-deployer download --repo unsloth/Qwen3.8-27B-GGUF --glob '*UD-Q3_K_XL.gguf'
cortex-deployer engines
cortex-deployer recipes
cortex-deployer render cortex_deployer/recipes/examples/qwen38-27b-q3-llamacpp.yaml --json
cortex-deployer connect --gateway wss://…/deployer/ws/register --token … --model … --upstream http://127.0.0.1:8080/v1
```

`connect` never invents a token.

## Repository fence

Development: Origin `shizuha-labs/cortex-deployer-beta`.
Public: GitHub `shizuha-labs/cortex-deployer` via leak-checked merge commits only.

See `CONTRIBUTING.md` and `SECURITY.md`.
