# Security

## What this repo must never contain

- Live pairing tokens, gateway tokens, API keys, cloud credentials
- Private hostnames, tailnet addresses, or first-party node inventory
- First-party fleet recipes (cluster fabric, qualification gates, workstation hostPaths)
- Placeholder or default pairing tokens (the CLI requires an explicit token)

A connect token is supplied at runtime (`--token` / `CORTEX_DEPLOYER_TOKEN`). It is not a Cortex `ProviderCredential`.

## Three-hop publication

1. **Origin `cortex-deployer-beta`** — internal SoT. Intermediate agent commits may be messy. History stays private.
2. **Origin `cortex-deployer`** — new-root history. Only a tree import + `merge --no-ff` after `scripts/leak-scan.sh` is green.
3. **GitHub `shizuha-labs/cortex-deployer`** — merge-only from hop 2, after a second scan. Not an ff-mirror of beta.

Never `git merge` a beta SHA into hop 2 or hop 3. That publishes beta history.

## Reporting

Email security@shizuha.com. Do not open a public issue for a live credential.
