#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

bash scripts/reproduce.sh
python validate.py --compare

mkdir -p tmp
cat > tmp/eval_manifest.json <<'JSON'
{
  "experiments": {
    "directed_tree_reconstruction": {
      "results": {
        "euclidean_mds": "outputs/euclidean_tree/predictions.npz",
        "finsler_mds": "outputs/finsler_tree/predictions.npz"
      }
    }
  }
}
JSON
python -m eval.tree_metrics --manifest tmp/eval_manifest.json --output tmp/manifest_scores.json
