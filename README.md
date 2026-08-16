# Cortex Deployer

Deploy an OpenAI-compatible model on your own machine (llama.cpp, SGLang, vLLM, or MLX) and **connect it to a Cortex router** so it appears on the catalog.

This repository is the leaf. The Cortex router, admission, billing, and first-party fleet stay in the private `cortex` repo.

## Install

```bash
pip install -e .
cortex-deployer --help
```

## Quick start

Serve a local engine, then dial out to a Cortex gateway (no inbound port on your machine):

```bash
# 1) start whatever you already run locally
#    llama-server / sglang / vllm / rapid-mlx — OpenAI /v1 on localhost

# 2) announce it to Cortex
cortex-deployer connect \
  --gateway wss://cortex.example.com/cortex/deployer/ws/register \
  --token "$CORTEX_DEPLOYER_TOKEN" \
  --model My-Model \
  --upstream http://127.0.0.1:8080/v1 \
  --alias my-model
```

`connect` never invents a default token and never hard-codes a model alias. The pairing token comes from the Cortex UI.

Render an engine command from a recipe without starting anything:

```bash
cortex-deployer render recipes/examples/llamacpp-cuda.yaml
cortex-deployer engines
```

## Engines

| Engine | Executor (v0.1) |
|---|---|
| `llamacpp` | process (`llama-server`) |
| `sglang` | process (`python -m sglang.launch_server`) |
| `vllm` | process (`python -m vllm.entrypoints.openai.api_server`) |
| `mlx` | process (`rapid-mlx serve` or `mlx_lm.server`) |

Unsupported topology (for example SGLang multi-node) fails closed at render time.

## Repository fence

Agent work lands on Origin **`shizuha-labs/cortex-deployer-beta`**.

A leak-checked **tree import + merge commit** (never `git merge <beta-sha>`, never force-push) promotes the same files onto Origin **`shizuha-labs/cortex-deployer`**.

GitHub `shizuha-labs/cortex-deployer` is a later merge-only hop from that fenced Origin repo. Beta commit objects must never be reachable from GitHub.

See `CONTRIBUTING.md` and `SECURITY.md`.
