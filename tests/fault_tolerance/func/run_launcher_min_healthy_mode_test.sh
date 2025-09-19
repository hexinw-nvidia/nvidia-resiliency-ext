# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# [DEPRECATED] This test verifies deprecation warning for --ft-restart-policy=min-healthy
# The min-healthy policy has been removed and --ft-restart-policy is deprecated
# To run the script: ./tests/fault_tolerance/func/run_launcher_min_healthy_mode_test.sh
# Expected result: TEST PASSED is printed and the exit code is 0

set -e

cleanup() {
    echo "Cleaning up before exit..."
    ( kill $(pgrep -f ft_launcher) > /dev/null 2>&1 || true )
    wait
}
trap cleanup EXIT

THIS_SCRIPT_DIR="$(dirname "$(realpath "$0")")"
WORKER_SCRIPT="${THIS_SCRIPT_DIR}/_launcher_mode_test_worker.py"

log_title() {
    echo
    echo "==============================================="
    echo "$1"
    echo "==============================================="
}

cd "${THIS_SCRIPT_DIR}/../../.."

# Kill any existing launcher processes
( kill $(pgrep -f ft_launcher) > /dev/null 2>&1 || true ) 

export LOGLEVEL='DEBUG'
COMMON_FT_ARGS="--ft-log-level=DEBUG"
COMMON_LAUNCHER_ARGS="--nproc-per-node=1 --nnodes=1:1 --rdzv-backend=c10d --rdzv_endpoint=localhost:12345 --max-restarts=0"
COMMON_TEST_SCRIPT_ARGS="--max-time=5"

log_title "Testing deprecation warning for --ft-restart-policy=min-healthy..."

# Capture stderr to check for deprecation warning
STDERR_OUTPUT=$(mktemp)

# Run ft_launcher with min-healthy policy and capture stderr
ft_launcher $COMMON_FT_ARGS $COMMON_LAUNCHER_ARGS --ft-restart-policy=min-healthy ${WORKER_SCRIPT} ${COMMON_TEST_SCRIPT_ARGS} 2>"$STDERR_OUTPUT" || true

# Check if deprecation warning is present in stderr
if grep -q "DeprecationWarning" "$STDERR_OUTPUT" && grep -q "ft-restart-policy.*deprecated" "$STDERR_OUTPUT"; then
    echo "✓ Deprecation warning detected successfully"
    echo "TEST PASSED"
    rm -f "$STDERR_OUTPUT"
    exit 0
else
    echo "✗ Deprecation warning not found in stderr"
    echo "stderr contents:"
    cat "$STDERR_OUTPUT"
    rm -f "$STDERR_OUTPUT"
    echo "TEST FAILED"
    exit 1
fi