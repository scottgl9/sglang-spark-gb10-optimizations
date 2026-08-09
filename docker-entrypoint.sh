#!/bin/bash
# docker-entrypoint.sh — launches sglang.sh with $SGLANG_PRESET (default:
# laguna-s-2.1) plus any extra args passed to `docker run`. Set SGLANG_PRESET
# to "build", "launch", or "shell" to run those sglang.sh subcommands instead
# of a model preset.
set -euo pipefail

cd /sgl-workspace/sglang
exec bash sglang.sh "${SGLANG_PRESET:-laguna-s-2.1}" "$@"
