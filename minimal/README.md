# nvidia-resiliency-ext-minimal

This directory holds metadata for the **`nvidia-resiliency-ext-minimal`** PyPI distribution: the same
`nvidia_resiliency_ext` package as the repository root, with **no** runtime MCP, LogSage, LangChain, or
**grpcio**. It **does** declare **`protobuf`** (the `google.protobuf` runtime used by generated
`*_pb2.py` and `cycle_info_writer`). The minimal wheel build uses **`grpcio-tools`** only at build time
to run `protoc`; it skips the CUPTI extension via `STRAGGLER_DET_SKIP_CUPTI_EXT_BUILD=1` in
`scripts/build_minimal_wheel.sh`.

**Default clone workflow** uses the parent directory: `pip install .` and `poetry build` at the
repo root produce the **full** `nvidia-resiliency-ext` package.

Poetry cannot package `../src` from this folder. Build the minimal wheel from the **repository root**:

```bash
./scripts/build_minimal_wheel.sh
```

Wheels are written to `minimal/dist/`. **gRPC** (`grpcio`) is optional for minimal; install the full package or add `grpcio` for gRPC log funnel / health-check gRPC paths.

**Protobuf version:** the minimal wheel is built with **grpcio-tools before 1.61** (pre-2024 gRPC Python). The embedded `protoc` emits stubs that expect a **`protobuf` PyPI package in the 4.21.x–5.x range** (see `minimal/pyproject.toml`). The **full** `nvidia-resiliency-ext` package may use a newer protoc floor and a higher `protobuf` bound.
