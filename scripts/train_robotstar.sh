#!/usr/bin/env bash
set -Eeuo pipefail
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m robotstar.train_generator "$@"
