from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


@dataclass(frozen=True)
class DDPRuntime:
    world_size: int
    rank: int
    local_rank: int
    is_ddp: bool
    is_main_process: bool


def get_ddp_runtime() -> DDPRuntime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_ddp = world_size > 1
    return DDPRuntime(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        is_ddp=is_ddp,
        is_main_process=(rank == 0),
    )


def setup_ddp_device(runtime: DDPRuntime) -> None:
    if runtime.is_ddp and torch.cuda.is_available():
        torch.cuda.set_device(int(runtime.local_rank))


def ddp_is_initialized(runtime: DDPRuntime) -> bool:
    return runtime.is_ddp and dist.is_available() and dist.is_initialized()


def ddp_barrier(runtime: DDPRuntime) -> None:
    if ddp_is_initialized(runtime):
        dist.barrier()


def all_gather_object(runtime: DDPRuntime, payload: Any) -> List[Any]:
    if not ddp_is_initialized(runtime):
        return [payload]
    gathered: List[Any] = [None for _ in range(int(runtime.world_size))]
    dist.all_gather_object(gathered, payload)
    return gathered


def gather_records_sorted_by_idx(runtime: DDPRuntime, local_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for part in all_gather_object(runtime, local_records):
        if part:
            merged.extend(part)
    merged.sort(key=lambda r: int(r.get("idx", -1)))
    return merged


def shard_with_global_indices(items: Sequence[T], runtime: DDPRuntime) -> List[Tuple[int, T]]:
    if not runtime.is_ddp:
        return list(enumerate(items))
    return [(idx, item) for idx, item in enumerate(items) if (idx % int(runtime.world_size)) == int(runtime.rank)]


def per_rank_eval_batch_size(eval_batch_size: int, runtime: DDPRuntime) -> int:
    batch_size = int(eval_batch_size)
    if batch_size <= 0:
        raise ValueError("eval_batch_size must be > 0")
    if not runtime.is_ddp:
        return batch_size
    if (batch_size % int(runtime.world_size)) != 0:
        raise ValueError(
            f"eval_batch_size ({batch_size}) must be divisible by WORLD_SIZE ({runtime.world_size}) in DDP mode."
        )
    return batch_size // int(runtime.world_size)

