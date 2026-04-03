#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Full wheel: repo root (poetry build). Minimal wheel: scripts/build_minimal_wheel.sh.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
poetry build -f wheel
"$ROOT/scripts/build_minimal_wheel.sh"
echo "Full wheel: $ROOT/dist/"
echo "Minimal wheel: $ROOT/minimal/dist/"
