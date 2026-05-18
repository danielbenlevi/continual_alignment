# ASR Summary

- Model key: `qwen_base`
- Model checkpoint name: `Qwen3-0.6B-Base`
- Run root: `/local/arise/db3651/continual_align/our_scripts/orchestrator_runs/qwen_base_updated_full_results`

## Method: ewcdr_base

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 11.28% ± 3.66% | 2.60% ± 1.64% | 7.28% ± 3.25% |
| After T2_gsm8k | 10.99% ± 5.10% | 2.56% ± 1.87% | 6.47% ± 2.87% |
| After T3_sst2 | 9.46% ± 2.20% | 10.54% ± 6.38% | 11.41% ± 5.82% |
| After T4_mbpp | 15.45% ± 6.02% | 31.28% ± 14.44% | 31.44% ± 14.47% |
| After T5_xsum | 14.20% ± 6.61% | 27.05% ± 8.32% | 27.28% ± 8.55% |
| After T6_sciq | 37.12% ± 7.95% | 84.01% ± 16.99% | 84.01% ± 16.99% |
| After T7_multiwoz | 44.94% ± 10.12% | 76.03% ± 9.68% | 76.03% ± 9.68% |

## Method: ewcdr_safety

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 12.50% ± 6.29% | 2.88% ± 2.59% | 7.56% ± 2.73% |
| After T2_gsm8k | 11.73% ± 6.15% | 2.47% ± 1.98% | 6.92% ± 3.26% |
| After T3_sst2 | 10.00% ± 5.42% | 11.22% ± 7.11% | 12.66% ± 6.75% |
| After T4_mbpp | 7.88% ± 3.60% | 12.66% ± 5.14% | 13.56% ± 4.94% |
| After T5_xsum | 5.48% ± 1.82% | 6.73% ± 1.90% | 7.12% ± 1.98% |
| After T6_sciq | 14.13% ± 12.98% | 42.85% ± 26.56% | 42.85% ± 26.56% |
| After T7_multiwoz | 14.17% ± 6.89% | 21.76% ± 11.69% | 21.83% ± 11.68% |

## Method: forever_base

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 9.52% ± 3.54% | 3.62% ± 1.19% | 6.51% ± 1.11% |
| After T2_gsm8k | 21.19% ± 5.98% | 14.39% ± 9.47% | 19.78% ± 8.24% |
| After T3_sst2 | 29.39% ± 7.70% | 66.76% ± 18.46% | 67.63% ± 18.02% |
| After T4_mbpp | 33.17% ± 11.05% | 38.33% ± 17.96% | 41.73% ± 16.91% |
| After T5_xsum | 25.61% ± 8.54% | 31.35% ± 11.96% | 34.58% ± 11.73% |
| After T6_sciq | 30.06% ± 3.88% | 40.48% ± 8.38% | 44.33% ± 7.44% |
| After T7_multiwoz | 28.33% ± 11.08% | 35.96% ± 22.30% | 40.32% ± 20.56% |

## Method: safety_forever_base

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 9.62% ± 4.04% | 3.81% ± 0.59% | 6.60% ± 2.23% |
| After T2_gsm8k | 16.99% ± 6.24% | 8.01% ± 3.18% | 12.82% ± 3.17% |
| After T3_sst2 | 28.11% ± 7.92% | 50.35% ± 18.10% | 52.76% ± 16.92% |
| After T4_mbpp | 22.76% ± 3.99% | 28.11% ± 10.24% | 31.99% ± 9.95% |
| After T5_xsum | 22.95% ± 4.32% | 21.86% ± 8.21% | 26.86% ± 6.69% |
| After T6_sciq | 23.78% ± 5.57% | 26.12% ± 10.77% | 29.39% ± 9.94% |
| After T7_multiwoz | 22.15% ± 7.53% | 28.27% ± 8.08% | 31.76% ± 8.20% |

## Method: safety_forever_v2_kl

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 9.74% ± 3.40% | 3.88% ± 0.80% | 6.63% ± 1.43% |
| After T2_gsm8k | 16.79% ± 5.37% | 10.42% ± 7.77% | 14.90% ± 6.28% |
| After T3_sst2 | 17.63% ± 9.66% | 27.12% ± 11.08% | 30.38% ± 11.65% |
| After T4_mbpp | 19.36% ± 6.53% | 18.21% ± 9.38% | 22.92% ± 8.71% |
| After T5_xsum | 18.27% ± 9.39% | 20.00% ± 15.31% | 23.37% ± 14.93% |
| After T6_sciq | 17.88% ± 8.44% | 24.84% ± 19.03% | 27.34% ± 18.31% |
| After T7_multiwoz | 16.41% ± 8.73% | 19.46% ± 12.12% | 22.98% ± 12.13% |

## Method: safety_forever_v2_layer_reg

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 9.55% ± 2.82% | 3.59% ± 0.95% | 6.25% ± 0.86% |
| After T2_gsm8k | 14.97% ± 5.42% | 9.39% ± 4.03% | 13.40% ± 4.50% |
| After T3_sst2 | 22.88% ± 8.42% | 43.43% ± 16.89% | 45.35% ± 17.01% |
| After T4_mbpp | 17.69% ± 6.10% | 23.17% ± 12.09% | 28.01% ± 11.37% |
| After T5_xsum | 22.08% ± 5.68% | 22.85% ± 11.04% | 27.31% ± 9.79% |
| After T6_sciq | 24.10% ± 6.49% | 27.05% ± 13.14% | 31.60% ± 12.99% |
| After T7_multiwoz | 25.48% ± 4.87% | 25.74% ± 6.54% | 30.87% ± 4.68% |
