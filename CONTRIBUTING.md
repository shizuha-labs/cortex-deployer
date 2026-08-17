# Contributing

## Where to work

Develop on Origin **`shizuha-labs/cortex-deployer-beta`** (`master`).

- Fleet agents: branch → PR → review → merge on **beta**
- Coordinator / framework-maintainer: commit on beta `master` (no force-push)

Do **not** open feature PRs on Origin `cortex-deployer` or on GitHub. Those repos only receive leak-checked merge commits.

## Public docs that must stay in lockstep

After any user-facing change, update **all** of:

- `README.md`, `docs/GETTING_STARTED.md`, `docs/EARN.md` (this repo)
- https://cortex.shizuha.com/deployer , `/deployer/install`, `/deployer/earn`
- wiki **How-to: Earn Hane with Cortex Deployer**

Do not link staff-only surfaces (`/cortex/backends`) from public docs.
Do not put pairing tokens, host inventory, or first-party recipes here.

## What belongs here

Engine launchers, recipes for public models, the outbound connect client, the hello/inventory/stream protocol, the local CLI.

Django, Cortex ORM, Kubernetes clients, admission fences, marketplace settlement, and first-party host recipes belong in private `cortex`.

The package must import with `DJANGO_SETTINGS_MODULE` unset.

## Promotion

```bash
# from the deploy repo, after leak-scan is green
deploy/k3s/origin/cortex-deployer-sync.sh import-beta
# later, after CTX-332:
# deploy/k3s/origin/cortex-deployer-sync.sh to-github
```

Never `git merge <beta-sha>`. Never `git push --force`.
