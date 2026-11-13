#!/usr/bin/env bash
# Runs the unit tests and every experiment, overwriting results/ in place.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python -m pytest -q
for exp in exp1 exp2 exp3 exp4 exp5 exp6; do
  echo "=== running exp.${exp} ==="
  python -m exp.${exp}
done
