import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import fire

from scripts_utils._clora_olora_common import train as _base_train


def train(*args, **kwargs):
    kwargs.setdefault("mode", "olora_safety")
    kwargs.setdefault("olora_safety_lambda_1", 2.5)
    kwargs.setdefault("olora_lambda_1", 0.5)
    kwargs.setdefault("learning_rate", 1e-3)
    return _base_train(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(train)
