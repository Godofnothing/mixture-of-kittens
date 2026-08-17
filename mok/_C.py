"""Select the MoK CUDA extension matching the GPU in use."""

from importlib import import_module

import torch


_extension = None


def _load_extension():
    global _extension
    if _extension is not None:
        return _extension

    if not torch.cuda.is_available():
        raise RuntimeError("MoK requires a CUDA GPU (SM100 or SM103)")

    capability = torch.cuda.get_device_capability()
    module_name = {
        (10, 0): "_C_sm100",
        (10, 3): "_C_sm103",
    }.get(capability)
    if module_name is None:
        raise RuntimeError(
            f"MoK requires an SM100 or SM103 GPU; found compute capability "
            f"{capability[0]}.{capability[1]}"
        )

    try:
        _extension = import_module(f".{module_name}", __package__)
    except ImportError as error:
        raise ImportError(
            f"MoK was not built for SM{capability[0]}{capability[1]}. "
            "Reinstall with MOK_ARCHS containing the required architecture."
        ) from error
    return _extension


def __getattr__(name):
    return getattr(_load_extension(), name)
