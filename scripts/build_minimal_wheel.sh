#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build nvidia-resiliency-ext-minimal wheel. Runs build.py (protoc via grpcio-tools); no CUPTI extension.
# *_pb2*.py are not committed — they are produced during this build into the temp tree.
#
# Usage: from repository root, ./scripts/build_minimal_wheel.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cp -a "$ROOT/src" "$TMP/src"
cp "$ROOT/build.py" "$TMP/build.py"
cp "$ROOT/README.md" "$TMP/README.md"
cp "$ROOT/minimal/pyproject.toml" "$TMP/pyproject.toml"

export STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1

if [[ -d "$ROOT/.git" ]]; then
  ln -s "$ROOT/.git" "$TMP/.git"
fi

mkdir -p "$ROOT/minimal/dist"
cd "$TMP"
poetry build -f wheel
shopt -s nullglob
for w in "$TMP/dist"/*.whl; do
  mv "$w" "$ROOT/minimal/dist/"
done
echo "Minimal wheel(s) -> $ROOT/minimal/dist/"
