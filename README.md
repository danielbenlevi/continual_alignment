# Toward Alignment Preservation in True Continual Learning Settings

**Note:** this repo is actively being developed. Code relevant to our COMS 6998 course project submission can be found at commit **dd5d008**.

To run all included methods, run this command from repo root (add CLI flags as needed):

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
- `clora_random`: CLoRA continual training with random regularization subspaces.
- `clora_safety`: CLoRA continual training with safety-aligned subspaces.
- `olora_standard`: O-LoRA continual training with standard orthogonality penalty.
- `olora_safety`: O-LoRA continual training with safety-weighted orthogonality penalty.
