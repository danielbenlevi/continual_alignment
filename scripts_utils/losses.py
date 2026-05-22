from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn

from scripts_utils.clora import CLoRALinear


def clora_regularization_loss(model: nn.Module) -> torch.Tensor:
    reg = None
    n = 0
    for m in model.modules():
        if isinstance(m, CLoRALinear):
            loss = m.clora_reg_loss()
            reg = loss if reg is None else (reg + loss)
            n += 1
    if reg is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    # Normalize by number of CLoRA modules so lambda has stable meaning.
    return reg / max(1, n)
