# Qwen3.8-27B on 16 GB VRAM — honest quant guidance (CTX-732)

Measured from the official `unsloth/Qwen3.8-27B-GGUF` tree (GGUF file sizes are
the **weights only**). llama.cpp additionally needs VRAM for the KV cache
(scales with context length), the CUDA compute buffers, and the vision
projector (`mmproj`, ~0.9 GB) when multimodal. On a 16 GB card (RTX 5080)
Windows also reserves ~0.5–1 GB for the driver, so the practical weight ceiling
is **~14 GB** if you want any usable context.

## GGUF size table (weights only)

| Quant | GGUF size | Fits 16 GB? | Honest note |
|---|---|---|---|
| `UD-IQ2_XXS` | 9.0 GB | ✅ Yes | Lowest quality; only for extreme VRAM pressure |
| `UD-IQ2_M` | 10.3 GB | ✅ Yes | Low quality; long-context option |
| `UD-Q2_K_XL` | 10.7 GB | ✅ Yes | Safe default (existing `qwen38-27b-q2-llamacpp.yaml`) |
| `UD-IQ3_XXS` | 11.9 GB | ✅ Yes | |
| `Q3_K_S` | 12.6 GB | ✅ Yes | |
| `UD-Q3_K_XL` | 13.4 GB | ⚠️ Tight | **Best quality that fits** — ~15 GB at 8k ctx; keep context ≤8k, 1 request |
| `Q3_K_M` | 13.8 GB | ⚠️ Tight | Same class as UD-Q3_K_XL; slightly larger |
| `IQ4_XS` | 15.7 GB | ⚠️ Very tight | Weights alone nearly fill the card — only with 4k ctx / reduced KV; risky |
| `Q4_K_S` | 16.1 GB | ❌ No | Exceeds 16 GB before KV cache |
| `Q4_0` | 16.1 GB | ❌ No | |
| `IQ4_NL` | 16.3 GB | ❌ No | |
| `Q4_K_M` | 17.1 GB | ❌ No | Needs ~19–20 GB with KV → **24 GB class** |
| `UD-Q4_K_XL` | 17.9 GB | ❌ No | Existing 24 GB recipe (`qwen38-27b-llamacpp.yaml`) |
| `Q5_K_M` | 19.8 GB | ❌ No | 24 GB class |
| `Q6_K` | 22.9 GB | ❌ No | 24 GB class |
| `Q8_0` | 29.0 GB | ❌ No | 32 GB+ class |

## Bottom line (do not overpromise)

- **On a 16 GB RTX 5080, the highest-quality 27B quant that honestly fits is
  `UD-Q3_K_XL` (13.4 GB) / `Q3_K_M` (13.8 GB) at ≤8k context.** Use the
  `qwen38-27b-q3-16gb-llamacpp.yaml` recipe.
- **`Q4_K_M` and above DO NOT fit 16 GB** — they OOM. Do not recommend them for
  a 5080; they belong on 24 GB cards (`qwen38-27b-llamacpp.yaml`, UD-Q4_K_XL).
- **`IQ4_XS` (15.7 GB) is a trap** — the weights nearly fill the card and leave
  no room for KV cache or context growth. Avoid unless you only ever use ~4k
  context.
- **Quality/throughput tradeoff:** Q3_K_XL is a real quality step up from Q2_K_XL
  (the previous 16 GB default) at the cost of ~2–3× tighter VRAM headroom and
  lower max context. If you need 32k+ context or multi-turn agent sessions,
  drop to Q2_K_XL or use a 9B/14B model instead.

## Smoke (pending CTX-717 host access)

Windows RTX 5080 (`asus-g700-1feggpfs`) smoke of the 16 GB recipe is queued
behind the v4 host access being provisioned under CTX-717. Smoke = server
starts, model loads, `/v1/chat/completions` returns a completion.
