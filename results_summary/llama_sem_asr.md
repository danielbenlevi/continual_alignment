# ASR Summary

- Model key: `llama`
- Model checkpoint name: `Llama-2-7b`
- Run root: `/local/arise/db3651/continual_align/our_scripts/orchestrator_runs/llama_7b_sem_results`

## Method: ewcdr_base

_Seeds aggregated: 3_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 73.78% ± 11.30% | 96.60% ± 2.48% | 96.67% ± 2.51% |
| After T2_gsm8k | 81.73% ± 2.77% | 99.42% ± 0.19% | 99.42% ± 0.19% |
| After T3_sst2 | 65.19% ± 7.02% | 99.94% ± 0.11% | 99.94% ± 0.11% |
| After T4_mbpp | 66.73% ± 14.24% | 100.00% ± 0.00% | 100.00% ± 0.00% |
| After T5_xsum | 56.15% ± 8.79% | 99.42% ± 0.58% | 99.42% ± 0.58% |
| After T6_sciq | 52.31% ± 16.40% | 100.00% ± 0.00% | 100.00% ± 0.00% |
| After T7_multiwoz | 55.00% ± 21.86% | 99.42% ± 0.84% | 99.42% ± 0.84% |

## Method: ewcdr_safety

_Seeds aggregated: 3_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 71.03% ± 1.68% | 94.29% ± 3.63% | 94.42% ± 3.59% |
| After T2_gsm8k | 80.83% ± 2.58% | 99.29% ± 0.48% | 99.29% ± 0.48% |
| After T3_sst2 | 71.15% ± 4.48% | 99.94% ± 0.11% | 99.94% ± 0.11% |
| After T4_mbpp | 76.73% ± 5.09% | 99.81% ± 0.19% | 99.81% ± 0.19% |
| After T5_xsum | 58.97% ± 22.79% | 99.94% ± 0.11% | 99.94% ± 0.11% |
| After T6_sciq | 57.63% ± 3.44% | 100.00% ± 0.00% | 100.00% ± 0.00% |
| After T7_multiwoz | 35.45% ± 20.24% | 100.00% ± 0.00% | 100.00% ± 0.00% |

## Method: forever_base

_Seeds aggregated: 3_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 64.17% ± 18.31% | 94.17% ± 4.11% | 94.36% ± 3.94% |
| After T2_gsm8k | 69.74% ± 7.63% | 96.35% ± 1.02% | 96.35% ± 1.02% |
| After T3_sst2 | 76.99% ± 2.74% | 98.01% ± 1.46% | 98.08% ± 1.35% |
| After T4_mbpp | 72.88% ± 1.17% | 96.73% ± 1.20% | 96.79% ± 1.31% |
| After T5_xsum | 63.59% ± 8.59% | 95.58% ± 1.50% | 95.90% ± 1.89% |
| After T6_sciq | 43.27% ± 9.00% | 86.92% ± 1.64% | 87.31% ± 1.35% |
| After T7_multiwoz | 44.36% ± 21.73% | 87.63% ± 10.55% | 88.40% ± 9.74% |

## Method: safety_forever_base

_Seeds aggregated: 3_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 62.44% ± 7.81% | 92.24% ± 1.25% | 92.44% ± 1.25% |
| After T2_gsm8k | 66.60% ± 22.37% | 97.24% ± 1.78% | 97.24% ± 1.78% |
| After T3_sst2 | 61.22% ± 4.32% | 93.72% ± 3.28% | 93.78% ± 3.39% |
| After T4_mbpp | 57.82% ± 8.33% | 96.15% ± 3.02% | 96.22% ± 3.01% |
| After T5_xsum | 64.74% ± 4.08% | 94.42% ± 5.50% | 94.55% ± 5.60% |
| After T6_sciq | 59.36% ± 4.18% | 94.74% ± 3.44% | 94.87% ± 3.42% |
| After T7_multiwoz | 54.10% ± 15.22% | 90.26% ± 14.06% | 90.45% ± 13.73% |

## Method: safety_forever_v2_kl

_Seeds aggregated: 3_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 43.53% ± 20.27% | 85.00% ± 8.38% | 85.51% ± 8.08% |
| After T2_gsm8k | 75.19% ± 10.95% | 95.51% ± 5.49% | 95.83% ± 4.95% |
| After T3_sst2 | 67.95% ± 14.74% | 96.92% ± 3.47% | 97.05% ± 3.25% |
| After T4_mbpp | 72.37% ± 4.71% | 97.69% ± 1.76% | 97.76% ± 1.66% |
| After T5_xsum | 68.78% ± 15.27% | 97.37% ± 2.39% | 97.37% ± 2.39% |
| After T6_sciq | 56.03% ± 8.76% | 91.47% ± 5.44% | 91.79% ± 4.89% |
| After T7_multiwoz | 74.42% ± 4.83% | 98.59% ± 1.16% | 98.59% ± 1.16% |

## Method: safety_forever_v2_layer_reg

_Seeds aggregated: 3_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 44.74% ± 16.28% | 83.78% ± 14.54% | 84.04% ± 14.10% |
| After T2_gsm8k | 61.15% ± 25.26% | 91.60% ± 8.35% | 92.05% ± 7.60% |
| After T3_sst2 | 67.12% ± 5.09% | 96.79% ± 3.42% | 97.12% ± 3.18% |
| After T4_mbpp | 61.54% ± 8.26% | 90.83% ± 8.96% | 91.86% ± 7.33% |
| After T5_xsum | 51.99% ± 15.15% | 88.14% ± 8.75% | 88.72% ± 8.27% |
| After T6_sciq | 65.77% ± 14.56% | 93.08% ± 10.50% | 93.85% ± 9.51% |
| After T7_multiwoz | 55.45% ± 20.90% | 91.47% ± 11.10% | 91.92% ± 10.50% |
