import gc
import os
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist

from mok.functional import clear_workspace_cache


@pytest.fixture(scope="session")
def context() -> Iterator[tuple[int, int, torch.device]]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    try:
        yield rank, world_size, device
    finally:
        dist.barrier()
        clear_workspace_cache()
        gc.collect()
        dist.destroy_process_group()
