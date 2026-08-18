# Cortex Deployer

Run open models on the GPU or Mac you already have. Chat locally, or connect
the same backend to [Cortex Router](https://cortex.shizuha.com/) and earn
**Hane** when someone else is routed through you.

Windows, Linux, and macOS. NVIDIA driver is the only extra host package for GPU.

**Official docs**

| Page | What it is |
| --- | --- |
| [cortex.shizuha.com/deployer](https://cortex.shizuha.com/deployer) | Live recipe catalog (VRAM × model × quant) |
| [cortex.shizuha.com/deployer/install](https://cortex.shizuha.com/deployer/install) | Install + update scripts |
| [cortex.shizuha.com/deployer/earn](https://cortex.shizuha.com/deployer/earn) | Earn Hane: install → serve → connect |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | First-run walkthrough |
| [docs/EARN.md](docs/EARN.md) | Provider path, pairing, what is billed |

## Install

Linux and macOS:

```bash
curl -fsSL https://cortex.shizuha.com/deployer/install.sh | bash
cortex-deployer server
# open http://127.0.0.1:7480  →  Choose a Qwen build
```

Windows PowerShell:

```powershell
irm https://cortex.shizuha.com/deployer/install.ps1 | iex
cortex-deployer server
```

The installer does **not** use the OS `pip` (Debian/Ubuntu PEP 668). It
bootstraps isolated `uv` + CPython under `~/.cortex-deployer` and puts a
wrapper on `~/.local/bin`. After the first install, **never reinstall** —
run `cortex-deployer update` or `cortex-deployer auto-update`. Models stay.
The local UI **Update Deployer** button does the same.

Linux/WSL listens on `0.0.0.0` (WSL eth0 + localhost). Windows defaults to
`127.0.0.1`. If port `7480` is busy, the next free high port is used.

From a checkout: `python -m cortex_deployer server`.

## First five minutes

1. Install (above) and run `cortex-deployer server`.
2. Open the printed URL (usually `http://127.0.0.1:7480`).
3. Either **Attach local server** (LM Studio / Ollama / vLLM already
   running — Deployer is only the tunnel) or **Choose a Qwen build**.
4. Chat in the same page, or point any OpenAI-compatible client at
   `http://127.0.0.1:7480/v1`.
5. To list on Cortex and earn Hane, follow
   [docs/EARN.md](docs/EARN.md).

Headless: `cortex-deployer setup` (recommended recipe) or
`cortex-deployer setup --recipe <name>`.

## Recommended pairings

The live picker is the source of truth. Typical full-GPU defaults:

| Card | Default pick |
| --- | --- |
| 8 GB | Qwen3-8B Q5 @ 32k (larger models listed as GPU+RAM offload) |
| 16 GB | Qwen3.5-9B UD-Q6 @ 64k. 27B only fills as Q2 @ 8k |
| 24 GB | Qwen3.8-27B Q4-class @ long context |
| Apple Silicon | MLX recipes from the same catalog |

Offload rows stay visible. `min_vram_mb` is the full-GPU number; offload
uses leftover layers in system RAM (`--fit`).

## What you get

| Surface | Role |
| --- | --- |
| Web UI `http://127.0.0.1:7480` | Setup, start/stop, GPU fit, download, chat, Cortex connect |
| `POST /api/setup` | Recommended recipe + engine + weights + start |
| `GET/POST /api/backends` | Same data as the UI |
| `GET /api/recommend` · `/api/downloads` | GPU-aware picker and Hugging Face pulls |
| `GET /v1/models` · `POST /v1/chat/completions` | Local OpenAI-compatible endpoint (CORS + stream) |
| `cortex-deployer connect` | Announce a running backend to Cortex Router |

Engines: **llama.cpp**, **SGLang**, **vLLM**, **MLX**.

Downloads do **not** require a Hugging Face token. The app tries Hugging
Face, then a public mirror. No shared token is baked in. Optional:
`HF_TOKEN` in the environment for private repos only.

State lives in `~/.cortex-deployer/` (Windows:
`%USERPROFILE%\.cortex-deployer\`). Override with `CORTEX_DEPLOYER_HOME`.

## Earn Hane

Hane is the Cortex inference currency. Consumers spend it on
`https://cortex.shizuha.com/v1`. Providers earn it when the router sends
someone else’s request to their listing. Take-rate is **0% at launch**.
Your own traffic to your own listing is unbilled.

```bash
# already running LM Studio / Ollama / vLLM on this box?
cortex-deployer attach --scan
cortex-deployer attach http://127.0.0.1:1234/v1

cortex-deployer connect \
  --gateway wss://cortex.shizuha.com/cortex/deployer/ws/register \
  --token <pairing from Cortex> \
  --model <served-name> \
  --upstream http://127.0.0.1:1234/v1
```

The local UI has a **Cortex** control on each running row. `connect`
never invents a token — pairing is issued when the listing is ready.
A backend stays off the public catalog until it answers health + a
canary.

Full path: [docs/EARN.md](docs/EARN.md) ·
[cortex.shizuha.com/deployer/earn](https://cortex.shizuha.com/deployer/earn).

## Update

```bash
cortex-deployer update
# or
curl -fsSL https://cortex.shizuha.com/deployer/update.sh | bash
cortex-deployer auto-update    # remember the preference, then upgrade
```

Windows: `irm https://cortex.shizuha.com/deployer/update.ps1 | iex`

## CLI

```bash
cortex-deployer server --host 127.0.0.1 --port 7480   # also: up, web
cortex-deployer update                                # in-place; keeps models
cortex-deployer update --check                        # exit 2 if a newer catalog exists
cortex-deployer update --restart                      # upgrade then exec server
cortex-deployer auto-update                           # persist upgrade-on-start, then update now
cortex-deployer auto-update --off                     # stop checking on start
cortex-deployer server --auto-update                  # same persist + apply if catalog is newer
cortex-deployer connect --auto-update …               # same, then re-exec connect only (engine stays)
# Idle Rapid-MLX D-METAL-CAP recycle (never while requests_running>0):
#   CORTEX_DEPLOYER_RECYCLE_CMD='launchctl kickstart -k gui/$UID/<engine-label>'
cortex-deployer attach --scan                         # find LM Studio / Ollama / vLLM
cortex-deployer attach http://127.0.0.1:1234/v1       # tunnel an existing local /v1
cortex-deployer setup                                 # recommended Qwen
cortex-deployer recommend
cortex-deployer recipes
cortex-deployer run cortex_deployer/recipes/examples/qwen38-27b-mlx.yaml
cortex-deployer connect --gateway wss://cortex.shizuha.com/cortex/deployer/ws/register \
  --token … --model … --upstream http://127.0.0.1:7480/v1
```

## Contributing and security

Development SoT is Origin `shizuha-labs/cortex-deployer-beta`. This public
tree is leak-checked and published from that hop — see
[CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
Report credentials to security@shizuha.com, not a public issue.
