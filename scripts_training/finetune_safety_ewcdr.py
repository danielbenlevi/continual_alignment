import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import fire

from scripts_training.finetune_ewcdr import train as _base_train


def train(*args, **kwargs):
    kwargs.setdefault("lamda", 30000.0)
    kwargs.setdefault("safety_task1_upweight", True)
    kwargs.setdefault("safety_lamda_multiplier", 0.5)
    return _base_train(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(train)
