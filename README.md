# Toward Alignment Preservation in True Continual Learning Settings

Run from repo root (add CLI flags as needed):

```bash
python full_experiment_orchestrator.py
```

Methods run:
- `forever_base`: baseline FOREVER continual LoRA training.
- `safety_forever_base`: doubled safety replay FOREVER variant.
- `safety_forever_v2_kl`: safety-FOREVER + safety token-level KL regularization.
- `safety_forever_v2_layer_reg`: safety-FOREVER + early-layer regularization boost.
- `ewcdr_base`: baseline EWC-DR continual regularization method.
- `ewcdr_safety`: EWC-DR with safety-focused Task-1 upweighting.
