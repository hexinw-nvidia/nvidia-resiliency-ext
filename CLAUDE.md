# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NVIDIA Resiliency Extension (NVRx)** is a PyTorch distributed training resiliency library. It provides fault tolerance (in-job restart without SLURM node reallocation), in-process restarting, async/local checkpointing, and straggler detection.

## Build and Install

```bash
# Install from source
pip install .

# Build wheels (Poetry)
poetry build -f wheel

# Build generates gRPC protobuf stubs and optionally a CUPTI C++ extension
# Skip CUPTI build if CUDA/CUPTI not available:
STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1 pip install .

# Minimal PyPI wheel (no runtime grpcio; build runs protoc and skips CUPTI):
./scripts/build_minimal_wheel.sh
```

The build process (`build.py`) always compiles three proto files (`nvhcd.proto`, `log_aggregation.proto`, `nvrx_interface.proto`) via `grpcio-tools` and optionally builds a pybind11 CUPTI extension. Generated `*_pb2*.py` / `*_pb2_grpc.py` / `*_pb2.pyi` are **gitignored** and must not be committed; `pip install .`, `poetry build`, and `./scripts/build_minimal_wheel.sh` regenerate them.

## Code Quality

```bash
# Format
black .
isort .

# Lint
ruff check .

# Check only (CI mode)
black --check .
isort --check-only .
ruff check .
```

Tool versions: `black==24.10.0`, `isort==5.13.2`, `ruff==0.6.9`. Line length is 100 chars. isort uses `profile = "black"`.

## Running Tests

```bash
# Run all unit tests in a module
pytest -s -vvv ./tests/fault_tolerance/unit/
pytest -s -vvv ./tests/inprocess/
pytest -s -vvv ./tests/checkpointing/unit/
pytest -s -vvv ./tests/ptl_resiliency/unit/
pytest -s -vvv ./tests/straggler/unit/
pytest -s -vvv ./tests/attribution/unit/

# Run a single test file
pytest -s -vvv ./tests/fault_tolerance/unit/test_rank_monitor.py

# Run a single test
pytest -s -vvv ./tests/fault_tolerance/unit/test_rank_monitor.py::TestClassName::test_method_name

# Straggler CPU-only subset (no CUPTI needed)
pytest -s -vvv tests/straggler/unit/ -k "test_all_gather_object_calls_num or test_fail_if_not_initialized"
```

Functional tests (`func/`) require a multi-GPU/SLURM environment and are not run in standard CI.

## Architecture

Source lives in `src/nvidia_resiliency_ext/`. Key modules:

### `fault_tolerance/`
In-job restart without reallocating SLURM nodes. Entry point: `ft_launcher` CLI (`launcher.py`). Extends PyTorch's elastic agent with rank monitoring via gRPC heartbeats. Key classes: `FaultToleranceConfig` (dataclass with all timeout/health settings), `RankMonitorServer`/`RankMonitorClient` (gRPC), `FtRendezvousBarrier` (custom rendezvous).

Section-based timeout model: user code calls `begin_section(name)` / `end_section(name)` to define named regions with individual timeouts. Out-of-section code uses a default timeout.

### `inprocess/`
Detect failures and restart within a single process (no job resubmission). Core pattern: wrap the user's training function with `Wrapper` or `Compose`. State machine managed in `state.py`. Background monitors (`monitor_thread.py`, `monitor_process.py`) detect hangs. `rank_assignment.py` handles dynamic rank reassignment on restart. States serialized with JSON (not pickle).

### `checkpointing/`
Two submodules:
- `async_ckpt/` — asynchronous checkpointing (save in background thread/process)
- `local/` — local-storage checkpointing for fast frequent saves

### `attribution/`
Log and trace analysis for failure attribution. Contains:
- `straggler/` — CUPTI-based GPU performance monitoring (`cupti_src/` is the C++ source). `straggler.py` exposes the main API. `reporting.py` handles statistical scoring and all-gather across ranks.
- `log_analyzer/`, `combined_log_fr/`, `trace_analyzer/` — LLM-based log analysis pipeline
- `mcp_integration/` — MCP (Model Context Protocol) server for AI log analysis

### `ptl_resiliency/`
PyTorch Lightning callbacks that wrap the above features:
- `FaultToleranceCallback` / `FaultToleranceSectionsCallback` — fault tolerance integration
- `StragglerDetectionCallback` — straggler detection
- `LocalCheckpointCallback` — local checkpointing

### `shared_utils/`
Shared infrastructure: `health_check.py` (GPU/NIC/storage health checks via NVML and gRPC), `log_manager.py` (structured logging), `grpc_log_server.py` + `log_aggregator.py` (centralized log collection), `memory.py` (GPU memory logging), `proto/` (protobuf definitions).

## Repository Root

### `services/`
Standalone services at the **repository root** (`services/`, not inside `src/nvidia_resiliency_ext/`): `nvrx_attrsvc/` (FastAPI server for LLM log analysis), `nvrx_smonsvc/` (SLURM job monitor).

## Key Environment Variables

| Variable | Purpose |
|---|---|
| `NVRX_LOG_DEBUG=1` | Enable debug logging |
| `STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1` | Skip CUPTI C++ build |
| `CUDA_PATH` | Override CUDA installation path (auto-detected otherwise) |

## Protobuf / gRPC

Three proto files define services:
- `nvhcd.proto` — Rank monitor heartbeat/health check service
- `log_aggregation.proto` — Centralized log collection
- `nvrx_interface.proto` — NVRx cycle info

Generated stubs are gitignored. If you modify `.proto` files, rebuild with `pip install .` or run `build.py` directly.

## Testing Notes

- CPU-only tests run without GPU. GPU/multi-node functional tests require real hardware.
- CI builds the full wheel from the repo root, runs `scripts/build_minimal_wheel.sh`, then copies minimal wheels into `dist/` (see `.github/workflows/unit_test.yml`); tests install the full `nvidia_resiliency_ext-*-cp*.whl` (not `*_minimal-*`).
- Some straggler tests require CUPTI; use `-k` filtering to select CPU-only tests.
- `MKL_SERVICE_FORCE_INTEL=1` may be needed to work around MKL threading issues in test environments.

## Contribution Workflow

- All changes must begin with a tracked issue, approved by NVRx engineers before code review starts.
- Contributions go through personal forks; do not push directly to upstream.
- Prefix PRs with `[WIP]` while under review.
- Commit format: `#<Issue Number> - <Commit Title>` (imperative mood, body optional).
- All commits must be GPG-signed (`git commit -S`) per the DCO requirement.
- New components require an accompanying README and at least one test.
- Build log must be clean (no warnings or errors) before submitting a PR.

## Documentation

Build and verify docs locally before submitting documentation changes:

```bash
pip install -U sphinx sphinx-rtd-theme sphinxcontrib-napoleon sphinx_copybutton lightning psutil defusedxml
sphinx-build -b html docs/source public/

# alternatively:
cd docs && make html
```

Output lands in `public/` or `docs/build/html/`. The build must complete without warnings or errors.
