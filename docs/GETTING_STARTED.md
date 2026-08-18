# Getting started

This is the first-run path for Cortex Deployer. Hardware pairings live on
the [live catalog](https://cortex.shizuha.com/deployer). Earning Hane is
documented in [EARN.md](EARN.md).

## 1. Install once

**Linux / macOS**

```bash
curl -fsSL https://cortex.shizuha.com/deployer/install.sh | bash
```

**Windows (PowerShell)**

```powershell
irm https://cortex.shizuha.com/deployer/install.ps1 | iex
```

NVIDIA Game Ready or Studio driver is the only extra host package for GPU.
No CUDA toolkit. No Hugging Face account.

The installer writes an isolated Python under `~/.cortex-deployer` and a
`cortex-deployer` wrapper on `~/.local/bin`. Add that directory to `PATH`
if the shell cannot find the command.

## 2. Start the local control plane

```bash
cortex-deployer server
```

Open the URL it prints (usually `http://127.0.0.1:7480`). If `7480` is
taken, the next free high port is used — trust the printed URL.

## 3. Attach a server you already run (optional)

If LM Studio, Ollama, vLLM, or llama.cpp is already answering on this
machine, do not start a second copy:

```bash
cortex-deployer attach --scan
cortex-deployer attach http://127.0.0.1:1234/v1
```

The local UI **Attach local server** button is the same. Then skip to
[EARN.md](EARN.md) to tunnel it to Cortex Router.

## 4. Choose a build

Click **Choose a Qwen build**. The picker:

- reads [catalog.json](https://cortex.shizuha.com/deployer/catalog.json)
- measures this machine’s VRAM
- marks one quant as the pick
- still lists offload options for larger models

Confirm. Deployer downloads official `llama-server` (or the engine for
that recipe) and the GGUF, then starts it.

Headless:

```bash
cortex-deployer setup
# or
cortex-deployer setup --recipe qwen35-9b-q6-llamacpp
```

## 5. Call it like OpenAI

```bash
curl http://127.0.0.1:7480/v1/models
curl http://127.0.0.1:7480/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<served-name>","messages":[{"role":"user","content":"hello"}]}'
```

Any OpenAI-compatible SDK works with `base_url=http://127.0.0.1:7480/v1`.
No API key is required for local traffic.

## 6. Stay current

```bash
cortex-deployer update          # keeps models
cortex-deployer auto-update     # upgrade on every start
```

Do not re-run the installer to upgrade. The UI **Update Deployer** button
is the same operation.

## Next

- Serve tokens on Cortex Router and earn Hane: [EARN.md](EARN.md)
- Recipe gallery: https://cortex.shizuha.com/deployer
