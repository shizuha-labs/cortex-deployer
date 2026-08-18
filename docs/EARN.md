# Earn Hane

Cortex is a two-sided inference market.

- **Consumers** spend Hane on `https://cortex.shizuha.com/v1` with a
  personal `sk-cortex-…` key.
- **Providers** run Cortex Deployer, connect a healthy backend, and the
  same ledger credits them when someone else is routed to that listing.

Take-rate is **0% at launch**. Traffic you send to your own listing is
unbilled. Hive reserved CPU/RAM is a separate Hane line.

Public walkthrough: https://cortex.shizuha.com/deployer/earn

## Path

Three ways in. Same Hane ledger.

**Already have a public HTTPS `/v1`?** Sign in at
https://cortex.shizuha.com/cortex/listings and paste it. Cortex probes
`/models` and a short completion. HTTPS only; localhost is rejected.

**Already serving locally (LM Studio / Ollama / vLLM / llama.cpp) and
no public URL?** Deployer is the tunnel. It does not start a second
engine — it attaches the one you have and opens an outbound WebSocket
to Cortex Router:

```bash
cortex-deployer attach --scan
cortex-deployer attach http://127.0.0.1:1234/v1          # LM Studio
# cortex-deployer attach http://127.0.0.1:11434/v1       # Ollama
# cortex-deployer attach http://127.0.0.1:8000/v1        # vLLM
```

Or open `http://127.0.0.1:7480` → **Attach local server**. Then
**Cortex** on that row (pairing token from the Cortex UI). `connect`
never invents a token.

**Need a local recipe?** Then:

1. **Install and serve locally** — [GETTING_STARTED.md](GETTING_STARTED.md).
   Chat on `127.0.0.1` first. If it does not answer `/v1/models` locally,
   it will not list.
2. **Sign in to Cortex** — https://cortex.shizuha.com/ → Inference key /
   Hane purse. New accounts open at https://shizuha.com/id/register
   (Turnstile required).
3. **Connect the backend.** The local UI has a Cortex control on each
   running row. Headless:

   ```bash
   cortex-deployer connect \
     --gateway wss://cortex.shizuha.com/cortex/deployer/ws/register \
     --token <pairing from Cortex> \
     --model <served-name> \
     --upstream http://127.0.0.1:7480/v1
   ```

   `connect` never invents a token. Pairing is issued when the listing is
   ready. Do not put tokens in recipes, git, or screenshots.
4. **Health + canary.** The listing stays off the public catalog until
   the backend answers liveness and a canary. Then it is routable and
   billable.
5. **Watch the purse.** https://cortex.shizuha.com/cortex/hane shows
   minted Hane, earned Hane, and spend.

## What is billed

| Traffic | Hane |
| --- | --- |
| Someone else routed to your listing | They spend; you earn |
| You calling your own listing | Unbilled |
| You calling anyone else’s listing | You spend |
| Daily mint / weekly faucet | Free starter balance |

Fiat numbers on the purse are a display suffix. Hane is the unit.

## Honest limits (launch)

- Pairing is not minted by the installer. Serve locally first; Cortex
  issues the token when the listing can be attached.
- Community listings are quality-scored. A backend that serves the
  wrong weights is delisted.
- Do not promise a specific rupee payout. Settlement rails are
  documented separately; Hane is the live unit today.

## Related

- Catalog: https://cortex.shizuha.com/deployer
- Install: https://cortex.shizuha.com/deployer/install
- Spend (OpenAI-compatible): https://cortex.shizuha.com/
