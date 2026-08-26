#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
cd "$ROOT"
if [[ -f PACKAGE_MANIFEST.sha256 ]]; then
  sha256sum -c PACKAGE_MANIFEST.sha256
fi
$PYTHON -m compileall -q robotstar tests
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" $PYTHON tests/test_representation.py
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" $PYTHON tests/test_tokenizer.py
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" $PYTHON tests/test_pyramid.py
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" $PYTHON tests/test_retrieval.py
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" $PYTHON tests/test_sampler.py
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" $PYTHON tests/test_scope.py
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" $PYTHON tests/test_attention_contract.py
for script in scripts/*.sh validate_install.sh; do bash -n "$script"; done
echo "[PASS] RobotSTAR release-candidate v1.1 validation complete"
