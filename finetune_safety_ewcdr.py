import fire

try:
    from our_scripts.finetune_ewcdr import train as _base_train
except ModuleNotFoundError as exc:
    if exc.name != "our_scripts":
        raise
    from finetune_ewcdr import train as _base_train


def train(*args, **kwargs):
    kwargs.setdefault("safety_task1_upweight", True)
    kwargs.setdefault("safety_lamda_multiplier", 1.5)
    return _base_train(*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(train)
