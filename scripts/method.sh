#!/bin/bash
# Register the paper's method (Finsler MDS: canonical Randers SMACOF, alpha 0.5,
# Isomap initialisation, uniform weights) with the evaluators and score it on
# the paper's exact trees.
#
# The evaluators run the registered command on every instance themselves, so
# registering is all a method needs to do.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

mkdir -p outputs/finsler_mds
cat > outputs/finsler_mds/method.json <<'JSON'
{
  "name": "finsler_mds",
  "command": ["python", "-m", "method.embed", "--alpha", "0.5", "--max-iter", "500", "--eps", "1e-5"],
  "description": "Finsler SMACOF in the 2D canonical Randers space (alpha 0.5), Isomap initialisation, uniform weights."
}
JSON

bash scripts/evaluate.sh
