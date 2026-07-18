# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import os
import sys
from pathlib import Path


def configure_therock_rocm_environment() -> None:
    """Match the collector's TheRock ROCm setup for focused ROCm/AITER tests."""
    os.environ.setdefault("VLLM_PLUGINS", "")
    os.environ.setdefault("ATOM_DISABLE_VLLM_PLUGIN", "1")

    _configure_therock_rocm_root()
    _expose_therock_amdsmi_python_package()
    _preload_torch_rocm_runtime()
    _reset_vllm_platform_cache()


def _configure_therock_rocm_root() -> None:
    try:
        from rocm_sdk import _devel
    except ImportError:
        return

    rocm_root = _devel.get_devel_root()
    if not (rocm_root / "lib" / "libamdhip64.so").exists():
        raise FileNotFoundError(f"TheRock ROCm devel root is incomplete: {rocm_root}")

    rocm_root_path = str(rocm_root)
    os.environ["ROCM_HOME"] = rocm_root_path
    os.environ["ROCM_PATH"] = rocm_root_path


def _expose_therock_amdsmi_python_package() -> None:
    try:
        import _rocm_sdk_core
    except ImportError:
        return

    amdsmi_parent = Path(_rocm_sdk_core.__file__).parent / "share" / "amd_smi"
    if not (amdsmi_parent / "amdsmi" / "__init__.py").exists():
        raise FileNotFoundError(f"TheRock AMDSMI package not found: {amdsmi_parent}")

    amdsmi_path = str(amdsmi_parent)
    if amdsmi_path not in sys.path:
        sys.path.insert(0, amdsmi_path)


def _preload_torch_rocm_runtime() -> None:
    try:
        import torch
    except ImportError:
        return

    libtorch_hip = Path(torch.__file__).parent / "lib" / "libtorch_hip.so"
    if libtorch_hip.exists():
        ctypes.CDLL(str(libtorch_hip), os.RTLD_GLOBAL | os.RTLD_NOW)


def _reset_vllm_platform_cache() -> None:
    # Pytest loads tests/conftest.py before test modules, and that can import
    # vLLM far enough to cache UnspecifiedPlatform before TheRock's bundled
    # amdsmi package is exposed.
    platforms_module = sys.modules.get("vllm.platforms")
    if platforms_module is not None:
        setattr(platforms_module, "_current_platform", None)
