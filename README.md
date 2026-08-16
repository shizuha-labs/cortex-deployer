# Cortex Deployer

Local control plane for open models. One process, a browser UI, and an OpenAI-compatible endpoint — the same idea as `npx @deepseek-ai/dsh web`, aimed at **managing inference backends** the way [Cortex Backends](https://cortex.shizuha.com/cortex/backends) does for the fleet.

Windows, Linux, and macOS.

```bash
curl -fsSL https://cortex.shizuha.com/deployer/install.sh | bash
cortex-deployer server
# Linux/WSL listens on 0.0.0.0 (WSL eth0 + localhost). Windows default is 127.0.0.1.
# open http://127.0.0.1:7480  →  Choose a Qwen build
```

The installer does **not** use the OS `pip` (Debian/Ubuntu PEP 668). It bootstraps an isolated `uv` + CPython under `~/.cortex-deployer` and puts a wrapper on `~/.local/bin`. After that, **never reinstall** — run `cortex-deployer update`, `cortex-deployer auto-update`, or `curl -fsSL https://cortex.shizuha.com/deployer/update.sh | bash`. Models stay. The UI **Update Deployer** button does the same.

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

16 GB cards: **Qwen3.5-9B Q6 @ 64k** is the agentic default. 27B only fits as Q2 at 8k; Q3/Q4 27B do not.

1. NVIDIA Game Ready / Studio driver (the only host install).
2. In PowerShell:

```powershell
irm https://cortex.shizuha.com/deployer/install.ps1 | iex
cortex-deployer server
```

3. Open http://127.0.0.1:7480 and click **Choose a Qwen build**.

The picker lists every catalog model that fits. On a 16 GB 5080 the recommended pick is **Qwen3.5-9B UD-Q6 @ 64k**; 27B Q2 is the only 27B option. Picking one:

- downloads official `llama.cpp` **Windows CUDA 13** + matching CUDA DLLs
- pulls that Unsloth GGUF
- starts `llama-server` and lists the served name at `/v1/models`

Chat in the same page, or point any OpenAI client at `http://127.0.0.1:7480/v1`. Headless: `cortex-deployer setup`.

Optional **Cortex** on a row announces it to a Cortex gateway (token from the Cortex UI). `connect` never invents a token.

State lives in `%USERPROFILE%\.cortex-deployer\` (override with `CORTEX_DEPLOYER_HOME`). Downloads do **not** ask for a Hugging Face token. The app tries Hugging Face, then a public mirror. No shared/default token is baked in (it would be extracted and banned). Optional: `HF_TOKEN` in the environment for private repos only.

## CLI

```bash
cortex-deployer server --host 127.0.0.1 --port 7480   # also: up, web
# if 7480 is excluded/busy (common on Windows), the next free high port is used
# llama-server defaults to 8080; if that port is excluded/busy the next free one is used
cortex-deployer update                                # in-place; keeps models
cortex-deployer update --check                        # exit 2 if a newer catalog release exists
cortex-deployer update --restart                      # upgrade then exec server
cortex-deployer auto-update                           # persist upgrade-on-start, then update now
cortex-deployer auto-update --off                     # stop checking on start
cortex-deployer server --auto-update                  # same persist + apply if catalog is newer
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
