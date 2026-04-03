# NVIDIA Resiliency Extension

The NVIDIA Resiliency Extension (NVRx) integrates multiple resiliency-focused solutions for PyTorch-based workloads. Users can modularly integrate NVRx capabilities into their own infrastructure to maximize AI training productivity at scale. NVRx maximizes goodput by enabling system-wide health checks, quickly detecting faults at runtime and resuming training automatically. NVRx minimizes loss of work by enabling fast and frequent checkpointing. 

For detailed documentation and usage information about each component, please refer to https://nvidia.github.io/nvidia-resiliency-ext/.

> ⚠️ NOTE: This project is still experimental and under active development. The code, features, and documentation are evolving rapidly. Please expect frequent updates and breaking changes. Contributions are welcome and we encourage you to watch for updates.

<img src="/docs/source/media/nvrx_core_features.png" alt="Figure highlighting core NVRx features including automatic restart, hierarchical checkpointing, fault detection and health checks" width="950" height="350">


## Core Components and Capabilities

- **[Fault Tolerance](https://nvidia.github.io/nvidia-resiliency-ext/fault_tolerance/index.html)**
  - Detection of hung ranks.  
  - Restarting training in-job, without the need to reallocate SLURM nodes.

- **[In-Process Restarting](https://nvidia.github.io/nvidia-resiliency-ext/inprocess/index.html)**
  - Detecting failures and enabling quick recovery.

- **[Async Checkpointing](https://nvidia.github.io/nvidia-resiliency-ext/checkpointing/async/index.html)**
  - Providing an efficient framework for asynchronous checkpointing.

- **[Local Checkpointing](https://nvidia.github.io/nvidia-resiliency-ext/checkpointing/local/index.html)**
  - Providing an efficient framework for local checkpointing.

- **[Straggler Detection](https://nvidia.github.io/nvidia-resiliency-ext/straggler_det/index.html)**
  - Monitoring GPU and CPU performance of ranks.  
  - Identifying slower ranks that may impede overall training efficiency.

- **Framework Integration**
  - Facilitating seamless [fault tolerance](https://nvidia.github.io/nvidia-resiliency-ext/fault_tolerance/integration/ptl.html) and [straggler detection](https://nvidia.github.io/nvidia-resiliency-ext/straggler_det/usage_guide.html#integration-guide) integration with PyTorch Lightning based workloads.
  - Providing integration with NVIDIA [NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/resiliency.html) framework, a scalable and cloud-native generative AI framework built for researchers and developers working on Large Language Models, Multimodal, and Speech AI (e.g. Automatic Speech Recognition and Text-to-Speech).

## Installation

NVRx is published as two PyPI distributions:

- **`nvidia-resiliency-ext`** (default) — Full stack: MCP, LogSage, LangChain NVIDIA endpoints, gRPC, and the `nvrx-mcp-analysis` console script, plus the same `nvidia_resiliency_ext` library as minimal.
- **`nvidia-resiliency-ext-minimal`** — Same import path (`import nvidia_resiliency_ext`) with a small dependency set (PyTorch, psutil, PyYAML, NVML, **protobuf**, etc.); **no** grpcio, MCP, LogSage, or LangChain. gRPC-only features still need the full package (or install **grpcio** yourself).

gRPC-based per-cycle log funneling and attribution features that need LogSage/MCP require the full package (or install those components yourself on top of minimal). The **full** package’s build uses current `grpcio-tools` and expects **`protobuf` ≥ 6.31.1** at runtime. **Minimal** uses pre-2024 `grpcio-tools` and a **4.x/5.x** `protobuf` range (see `minimal/README.md`).

### From PyPI

- Full stack: `pip install nvidia-resiliency-ext`
- Minimal only: `pip install nvidia-resiliency-ext-minimal`

Pre-built wheels avoid compiling protobuf extensions at install time where possible; building from an sdist still uses the build backend in `pyproject.toml`.

### From sources (this repository)

From the **repository root** (default — full package, same as PyPI `nvidia-resiliency-ext`):

- `git clone https://github.com/NVIDIA/nvidia-resiliency-ext`
- `cd nvidia-resiliency-ext`
- `pip install .` or `poetry install` / `poetry build` (these run `build.py`, which generates `*_pb2*.py` / `*_pb2_grpc.py` / `*_pb2.pyi` next to the `.proto` files; those outputs are **gitignored** — do not commit them. Run `python clean.py` to remove generated protos and `build/` / `dist/`.)

Minimal wheel (no grpcio in metadata; see `minimal/README.md`):

- `./scripts/build_minimal_wheel.sh` (writes `minimal/dist/`)

To build **both** wheels: `scripts/build_both_wheels.sh`.

### Platform Support

| Category             | Supported Versions / Requirements                                          |
|----------------------|----------------------------------------------------------------------------|
| Architecture         | x86_64, arm64                                                              |
| Operating System     | Ubuntu 22.04, 24.04                                                        |
| Python Version       | >= 3.10, < 3.13                                                            |
| PyTorch Version      | >= 2.5.1, >= 2.8.0 (Fault Attribution)                                      |
| CUDA & CUDA Toolkit  | >= 12.8                                                                    |
| NVML Driver          | >= 535 (570 required for GPU health check)                                 |
| NCCL Version         | < 2.28.3 OR >= 2.28.9 (avoid NCCL 2.28.3–2.28.8 due to inprocess issue)    |
| TE Version           | >= 2.5                                                                     |

