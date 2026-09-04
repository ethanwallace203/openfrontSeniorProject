#!/usr/bin/env bash
# Thin CLI entry point — all logic lives in engine/bridge/src/replayGame.ts.
# See scripts/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
npm run replay -- "$@"
