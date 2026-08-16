#!/usr/bin/env bash
# Fail if the working tree looks like it still has live secrets or first-party
# fleet inventory. Run before every import into Origin cortex-deployer / GitHub.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

# Patterns that must never ship. Keep the list boring and high-precision —
# a noisy scanner gets ignored.
patterns=(
  'gateway-tok-change-me'
  'BEGIN (RSA|OPENSSH|EC) PRIVATE KEY'
  'sk-[A-Za-z0-9]{20,}'
  'hf_[A-Za-z0-9]{20,}'
  'AKIA[0-9A-Z]{16}'
  'ghp_[A-Za-z0-9]{20,}'
  'gho_[A-Za-z0-9]{20,}'
  '100\.64\.0\.'
  'i9-ws'
  'gx10-[0-9]'
  '/home/phoenix/'
  '/Users/shizuha-mb/'
  'shizuha-mb@'
  'FORGEJO_TOKEN='
  'password: ["'\''][^"'\'']+["'\'']'
)

fail=0
for pat in "${patterns[@]}"; do
  # Exclude this scanner, license boilerplate, and git metadata.
  hits="$(
    grep -RInE --binary-files=without-match \
      --exclude-dir=.git \
      --exclude-dir=.venv \
      --exclude-dir=__pycache__ \
      --exclude='leak-scan.sh' \
      --exclude='LICENSE' \
      -e "$pat" . || true
  )"
  if [[ -n "$hits" ]]; then
    echo "LEAK SCAN FAIL pattern=$pat"
    echo "$hits"
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "leak-scan: refused"
  exit 1
fi
echo "leak-scan: clean"
