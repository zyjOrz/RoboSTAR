#!/usr/bin/env bash
set -Eeuo pipefail
python -m robotstar.build_cache "$@"
