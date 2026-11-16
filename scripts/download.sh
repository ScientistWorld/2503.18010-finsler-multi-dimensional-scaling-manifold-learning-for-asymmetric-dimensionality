#!/bin/bash
# Download all data needed for this environment.
#
# Run on a node with internet access BEFORE building the container or
# submitting compute jobs. Idempotent: files that exist with the right
# checksum are skipped.
#
# Estimated total download size: < 1 MB
# Estimated disk usage after extraction: < 1 MB
#
# What is fetched:
#   data/downloads/cora/cora.cites          LINQS Cora citation graph (2,708 papers, 5,429 citations)
#   data/downloads/citeseer/citeseer.cites  LINQS CiteSeer citation graph (4,732 citation lines)
# These are the raw files behind the processed edge lists in
# data/citation_graphs/ (and the evaluators' own copies in eval/*/graphs/),
# which are tracked in git and are rebuilt/checked with
#   python data/build_citation_graphs.py [--check]
# The directed-tree instances are generated procedurally and need no download.

set -e

cd /home/user

CORA_SHA=ec1a372391b7f0f60a6aff0084e8abd8f19f0faa7e1f2441a41c492042d5945e
CITESEER_SHA=cd9730630eaaa745058c2cd2cce6b9afb79ca71278a479aebec55f4cf5eb7f30

have() {  # have <file> <sha256>
  [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]
}

fetch() {  # fetch <url> <dest>; LINQS serves an incomplete TLS chain on some
           # hosts, so retry without verification (the sha256 check below
           # guards the content).
  curl -fsSL --retry 3 -o "$2" "$1" || curl -fsSL --retry 3 --insecure -o "$2" "$1"
}

mkdir -p data/downloads/cora data/downloads/citeseer
TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT

if ! have data/downloads/cora/cora.cites "$CORA_SHA"; then
  echo "Fetching Cora citations"
  # Kipf's pygcn repository mirrors the LINQS file byte for byte; fall back to LINQS.
  if ! fetch https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/cora.cites "$TMPD/cora.cites" \
     || [ "$(sha256sum "$TMPD/cora.cites" | cut -d' ' -f1)" != "$CORA_SHA" ]; then
    fetch https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz "$TMPD/cora.tgz"
    tar -xzf "$TMPD/cora.tgz" -C "$TMPD" cora/cora.cites
    mv "$TMPD/cora/cora.cites" "$TMPD/cora.cites"
  fi
  mv "$TMPD/cora.cites" data/downloads/cora/cora.cites
fi
have data/downloads/cora/cora.cites "$CORA_SHA" || { echo "cora.cites checksum mismatch" >&2; exit 1; }

if ! have data/downloads/citeseer/citeseer.cites "$CITESEER_SHA"; then
  echo "Fetching CiteSeer citations"
  fetch https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz "$TMPD/citeseer.tgz"
  tar -xzf "$TMPD/citeseer.tgz" -C "$TMPD" citeseer/citeseer.cites
  mv "$TMPD/citeseer/citeseer.cites" data/downloads/citeseer/citeseer.cites
fi
have data/downloads/citeseer/citeseer.cites "$CITESEER_SHA" || { echo "citeseer.cites checksum mismatch" >&2; exit 1; }

# The tracked processed graphs must agree with the raw files.
python3 data/build_citation_graphs.py --check
echo "Downloads complete."
