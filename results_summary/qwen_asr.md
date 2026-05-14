# ASR Summary

- Model key: `qwen`
- Model checkpoint name: `Qwen3-0.6B`
- Run root: `/local/arise/db3651/continual_align/our_scripts/orchestrator_runs/updated_full_results`

## Method: ewcdr_base

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 15.06% ± 2.92% | 1.86% ± 1.67% | 7.92% ± 1.53% |
| After T2_gsm8k | 7.21% ± 1.50% | 1.25% ± 0.94% | 4.62% ± 1.23% |
| After T3_sst2 | 23.08% ± 10.50% | 38.56% ± 21.33% | 38.94% ± 20.98% |
| After T4_mbpp | 26.19% ± 11.07% | 51.54% ± 24.77% | 51.57% ± 24.71% |
| After T5_xsum | 15.90% ± 5.02% | 45.61% ± 25.13% | 45.61% ± 25.13% |
| After T6_sciq | 39.81% ± 11.45% | 97.44% ± 2.53% | 97.44% ± 2.53% |
| After T7_multiwoz | 40.93% ± 14.71% | 86.54% ± 5.48% | 86.54% ± 5.48% |

## Method: ewcdr_safety

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 14.94% ± 2.73% | 1.86% ± 1.78% | 7.56% ± 1.35% |
| After T2_gsm8k | 9.33% ± 2.43% | 1.70% ± 1.27% | 5.45% ± 1.45% |
| After T3_sst2 | 11.41% ± 5.54% | 15.77% ± 15.16% | 17.60% ± 14.62% |
| After T4_mbpp | 16.51% ± 9.98% | 29.29% ± 22.44% | 30.35% ± 21.37% |
| After T5_xsum | 5.35% ± 5.21% | 9.78% ± 11.86% | 10.06% ± 11.62% |
| After T6_sciq | 44.07% ± 9.52% | 90.64% ± 6.70% | 90.64% ± 6.70% |
| After T7_multiwoz | 30.64% ± 19.38% | 53.97% ± 23.73% | 54.17% ± 23.59% |

## Method: forever_base

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 14.39% ± 2.48% | 1.57% ± 1.13% | 7.08% ± 1.60% |
| After T2_gsm8k | 9.17% ± 3.62% | 4.23% ± 4.60% | 7.34% ± 4.57% |
| After T3_sst2 | 24.17% ± 7.83% | 60.03% ± 8.78% | 60.87% ± 8.66% |
| After T4_mbpp | 20.32% ± 9.39% | 35.99% ± 16.03% | 37.31% ± 16.15% |
| After T5_xsum | 17.05% ± 6.79% | 23.78% ± 10.21% | 26.47% ± 9.80% |
| After T6_sciq | 22.15% ± 9.33% | 42.85% ± 21.70% | 44.26% ± 21.09% |
| After T7_multiwoz | 20.45% ± 6.34% | 34.17% ± 15.24% | 36.19% ± 14.34% |

## Method: safety_forever_base

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 15.64% ± 3.58% | 2.28% ± 1.57% | 7.85% ± 2.09% |
| After T2_gsm8k | 12.95% ± 6.12% | 2.02% ± 2.26% | 8.46% ± 4.63% |
| After T3_sst2 | 19.84% ± 8.68% | 42.34% ± 22.01% | 44.62% ± 21.00% |
| After T4_mbpp | 20.93% ± 6.20% | 21.76% ± 12.03% | 26.28% ± 10.44% |
| After T5_xsum | 17.02% ± 6.07% | 18.01% ± 10.77% | 21.89% ± 9.58% |
| After T6_sciq | 17.47% ± 4.75% | 25.61% ± 9.67% | 28.04% ± 9.76% |
| After T7_multiwoz | 17.79% ± 4.29% | 21.92% ± 10.97% | 25.16% ± 10.67% |

## Method: safety_forever_v2_kl

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 15.54% ± 4.44% | 1.79% ± 1.33% | 7.53% ± 1.89% |
| After T2_gsm8k | 12.05% ± 7.69% | 2.66% ± 5.22% | 7.56% ± 6.23% |
| After T3_sst2 | 15.83% ± 7.36% | 31.09% ± 15.47% | 32.37% ± 15.94% |
| After T4_mbpp | 19.42% ± 8.49% | 18.11% ± 7.38% | 22.44% ± 8.34% |
| After T5_xsum | 15.51% ± 7.11% | 13.43% ± 7.89% | 16.86% ± 6.26% |
| After T6_sciq | 18.08% ± 7.16% | 22.24% ± 13.78% | 24.97% ± 13.07% |
| After T7_multiwoz | 17.40% ± 7.68% | 18.40% ± 12.35% | 21.38% ± 12.04% |

## Method: safety_forever_v2_layer_reg

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 15.22% ± 3.18% | 1.60% ± 1.09% | 7.12% ± 1.65% |
| After T2_gsm8k | 13.33% ± 6.03% | 2.85% ± 2.81% | 7.85% ± 3.00% |
| After T3_sst2 | 20.35% ± 7.84% | 50.51% ± 19.62% | 52.12% ± 18.66% |
| After T4_mbpp | 20.71% ± 5.03% | 28.94% ± 15.12% | 32.37% ± 13.74% |
| After T5_xsum | 21.22% ± 8.31% | 26.86% ± 15.12% | 30.38% ± 14.27% |
| After T6_sciq | 22.34% ± 6.96% | 29.17% ± 13.58% | 32.12% ± 12.70% |
| After T7_multiwoz | 20.67% ± 8.09% | 25.16% ± 12.28% | 27.85% ± 13.30% |
