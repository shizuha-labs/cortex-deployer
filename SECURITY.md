# Security

## What this repo must never contain

- Live pairing tokens, gateway tokens, API keys, cloud credentials
- Private hostnames, tailnet addresses, or first-party node inventory
- First-party fleet recipes (cluster fabric, qualification gates, workstation hostPaths)
- Placeholder or default pairing tokens (the CLI requires an explicit token)

A connect token is supplied at runtime (`--token` / `CORTEX_DEPLOYER_TOKEN`). It is not a Cortex `ProviderCredential`.

## Two-lane publication (2026-09-22)

1. **Origin `cortex-deployer` `master`** — dev lane / internal SoT. Intermediate agent commits may be messy. Dev-only history stays private.
2. **Origin `cortex-deployer` `publish`** — publish lane, driven by an agent: tree-only imports (`git commit-tree master^{tree}`) gated by `cortex-deployer-publish-guard.sh` (DAG invariant: no dev-only commit may enter the publish chain) plus a leak scan.
3. **GitHub `shizuha-labs/cortex-deployer` `main`** — fast-forward-only push of hop 2 after CI is green. Never force-pushed.

Never `git merge` a dev SHA into `publish`. That publishes dev history. (The legacy three-hop flow via `cortex-deployer-beta` / `cortex-deployer-deprecated` was retired 2026-09-22.)

## Reporting

Email security@shizuha.com. Do not open a public issue for a live credential.
