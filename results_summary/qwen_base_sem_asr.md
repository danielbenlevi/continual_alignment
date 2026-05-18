# ASR Summary

- Model key: `qwen_base`
- Model checkpoint name: `Qwen3-0.6B-Base`
- Run root: `/local/arise/db3651/continual_align/our_scripts/orchestrator_runs/qwen_base_sem_results`

## Method: ewcdr_base

_Seeds aggregated: 3_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 69.81% ± 7.02% | 94.55% ± 1.57% | 94.62% ± 1.53% |
| After T2_gsm8k | 79.36% ± 1.28% | 96.54% ± 2.22% | 96.54% ± 2.22% |
| After T3_sst2 | 75.64% ± 2.80% | 97.56% ± 0.95% | 97.56% ± 0.95% |
| After T4_mbpp | 76.35% ± 1.95% | 97.31% ± 1.02% | 97.44% ± 0.95% |
| After T5_xsum | 79.36% ± 1.44% | 97.31% ± 1.07% | 97.31% ± 1.07% |
| After T6_sciq | 70.71% ± 4.74% | 97.76% ± 1.55% | 97.76% ± 1.55% |
| After T7_multiwoz | 80.90% ± 1.74% | 98.91% ± 0.40% | 98.97% ± 0.29% |

## Method: ewcdr_safety

_Seeds aggregated: 3_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 69.74% ± 6.95% | 94.55% ± 1.57% | 94.62% ± 1.53% |
| After T2_gsm8k | 79.74% ± 2.33% | 96.54% ± 1.95% | 96.54% ± 1.95% |
| After T3_sst2 | 75.71% ± 1.09% | 98.40% ± 1.61% | 98.40% ± 1.61% |
| After T4_mbpp | 76.22% ± 1.61% | 97.12% ± 1.68% | 97.18% ± 1.72% |
| After T5_xsum | 79.94% ± 0.44% | 96.92% ± 1.83% | 96.92% ± 1.83% |
| After T6_sciq | 75.71% ± 4.99% | 97.31% ± 1.71% | 97.37% ± 1.60% |
| After T7_multiwoz | 80.90% ± 1.28% | 97.82% ± 1.31% | 97.82% ± 1.31% |

## Method: forever_base

_Seeds aggregated: 3_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 74.55% ± 4.63% | 96.09% ± 1.83% | 96.09% ± 1.83% |
| After T2_gsm8k | 65.90% ± 5.99% | 94.23% ± 3.24% | 94.42% ± 3.20% |
| After T3_sst2 | 71.73% ± 3.74% | 97.56% ± 1.97% | 97.76% ± 1.97% |
| After T4_mbpp | 67.12% ± 8.12% | 96.54% ± 1.95% | 96.60% ± 1.87% |
| After T5_xsum | 64.68% ± 9.61% | 95.19% ± 1.45% | 95.19% ± 1.45% |
| After T6_sciq | 62.95% ± 2.56% | 94.68% ± 1.55% | 94.81% ± 1.54% |
| After T7_multiwoz | 63.33% ± 4.94% | 94.04% ± 2.91% | 94.29% ± 2.75% |

## Method: safety_forever_base

_Seeds aggregated: 3_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 74.62% ± 4.70% | 96.09% ± 1.83% | 96.09% ± 1.83% |
| After T2_gsm8k | 65.58% ± 8.36% | 93.40% ± 3.61% | 93.46% ± 3.59% |
| After T3_sst2 | 70.90% ± 3.74% | 97.50% ± 1.92% | 97.56% ± 1.83% |
| After T4_mbpp | 68.97% ± 5.11% | 95.58% ± 2.17% | 95.71% ± 2.06% |
| After T5_xsum | 70.90% ± 5.78% | 96.22% ± 2.39% | 96.22% ± 2.39% |
| After T6_sciq | 69.55% ± 5.34% | 96.15% ± 2.50% | 96.28% ± 2.61% |
| After T7_multiwoz | 66.99% ± 5.65% | 97.05% ± 1.28% | 97.12% ± 1.17% |

## Method: safety_forever_v2_kl

_Seeds aggregated: 3_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 74.68% ± 4.59% | 96.09% ± 1.83% | 96.09% ± 1.83% |
| After T2_gsm8k | 64.94% ± 7.28% | 91.60% ± 4.84% | 91.73% ± 4.62% |
| After T3_sst2 | 72.82% ± 3.82% | 97.69% ± 1.35% | 97.69% ± 1.35% |
| After T4_mbpp | 68.97% ± 5.56% | 95.77% ± 0.84% | 95.90% ± 0.80% |
| After T5_xsum | 67.18% ± 1.72% | 97.12% ± 1.67% | 97.18% ± 1.55% |
| After T6_sciq | 68.46% ± 3.20% | 95.32% ± 1.66% | 95.32% ± 1.66% |
| After T7_multiwoz | 65.26% ± 1.60% | 95.58% ± 0.96% | 95.58% ± 0.96% |

## Method: safety_forever_v2_layer_reg

_Seeds aggregated: 3_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 74.68% ± 4.59% | 96.09% ± 1.83% | 96.09% ± 1.83% |
| After T2_gsm8k | 67.76% ± 6.13% | 94.23% ± 3.17% | 94.23% ± 3.17% |
| After T3_sst2 | 72.44% ± 3.87% | 98.21% ± 0.29% | 98.40% ± 0.22% |
| After T4_mbpp | 67.18% ± 8.20% | 95.77% ± 1.17% | 95.90% ± 1.11% |
| After T5_xsum | 65.90% ± 1.54% | 96.54% ± 0.77% | 96.73% ± 0.88% |
| After T6_sciq | 59.68% ± 4.51% | 96.92% ± 1.85% | 96.92% ± 1.85% |
| After T7_multiwoz | 56.60% ± 2.51% | 94.55% ± 1.31% | 94.68% ± 1.28% |
