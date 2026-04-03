# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for optional dependencies (full install vs minimal wheel)."""

from typing import Optional

FULL_INSTALL_PIP_HINT = (
    "Install the full NVIDIA Resiliency Extension (MCP, LogSage, gRPC, etc.): "
    "pip install nvidia-resiliency-ext"
)


def raise_full_install_import_error(feature: str, cause: Optional[BaseException] = None) -> None:
    """Raise ImportError directing users to the full PyPI distribution."""
    msg = f"{feature} requires optional dependencies. {FULL_INSTALL_PIP_HINT}"
    raise ImportError(msg) from cause
