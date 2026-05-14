# ASR Summary

- Model key: `llama`
- Model checkpoint name: `Llama-3.2-3B-Instruct`
- Run root: `/local/arise/db3651/continual_align/our_scripts/orchestrator_runs/llama_updated_full_results`

## Method: ewcdr_base

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 14.39% ± 4.37% | 2.24% ± 1.89% | 8.04% ± 3.22% |
| After T2_gsm8k | 24.07% ± 15.45% | 29.04% ± 17.84% | 29.36% ± 18.02% |
| After T3_sst2 | 11.86% ± 9.69% | 39.01% ± 14.81% | 39.07% ± 14.88% |
| After T4_mbpp | 12.56% ± 5.90% | 36.03% ± 16.87% | 36.03% ± 16.87% |
| After T5_xsum | 15.64% ± 6.25% | 38.75% ± 10.27% | 38.75% ± 10.27% |
| After T6_sciq | 27.88% ± 8.27% | 70.90% ± 19.76% | 70.93% ± 19.69% |
| After T7_multiwoz | 42.82% ± 11.82% | 75.90% ± 16.36% | 75.93% ± 16.31% |

## Method: ewcdr_safety

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 12.88% ± 3.45% | 2.24% ± 1.43% | 6.35% ± 2.28% |
| After T2_gsm8k | 25.16% ± 8.72% | 39.20% ± 18.58% | 40.03% ± 17.49% |
| After T3_sst2 | 13.65% ± 11.18% | 39.55% ± 20.72% | 39.55% ± 20.72% |
| After T4_mbpp | 16.38% ± 11.21% | 45.45% ± 18.30% | 45.64% ± 18.43% |
| After T5_xsum | 11.92% ± 8.37% | 28.43% ± 15.13% | 28.43% ± 15.13% |
| After T6_sciq | 28.53% ± 9.24% | 77.60% ± 14.09% | 77.60% ± 14.09% |
| After T7_multiwoz | 22.98% ± 5.65% | 57.69% ± 19.48% | 57.69% ± 19.48% |

## Method: forever_base

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 9.17% ± 3.50% | 2.66% ± 1.22% | 5.38% ± 2.46% |
| After T2_gsm8k | 6.92% ± 4.66% | 3.94% ± 2.24% | 5.93% ± 4.13% |
| After T3_sst2 | 15.42% ± 3.79% | 8.69% ± 1.43% | 14.29% ± 3.18% |
| After T4_mbpp | 15.90% ± 3.99% | 5.83% ± 3.28% | 12.08% ± 3.92% |
| After T5_xsum | 14.20% ± 2.86% | 3.94% ± 1.96% | 9.65% ± 2.77% |
| After T6_sciq | 8.97% ± 4.76% | 6.96% ± 6.26% | 9.71% ± 7.07% |
| After T7_multiwoz | 15.03% ± 6.80% | 6.22% ± 3.32% | 11.38% ± 5.43% |

## Method: safety_forever_base

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 11.28% ± 3.78% | 2.02% ± 0.50% | 6.63% ± 1.72% |
| After T2_gsm8k | 10.00% ± 8.19% | 3.30% ± 3.69% | 6.22% ± 4.35% |
| After T3_sst2 | 14.58% ± 4.27% | 9.33% ± 2.39% | 12.37% ± 1.58% |
| After T4_mbpp | 16.79% ± 6.55% | 7.85% ± 2.10% | 13.85% ± 4.29% |
| After T5_xsum | 13.33% ± 7.00% | 5.03% ± 2.50% | 10.64% ± 2.04% |
| After T6_sciq | 9.87% ± 4.91% | 5.96% ± 2.74% | 9.29% ± 3.58% |
| After T7_multiwoz | 10.67% ± 3.64% | 5.87% ± 2.81% | 9.42% ± 2.08% |

## Method: safety_forever_v2_kl

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 11.89% ± 3.00% | 2.15% ± 0.80% | 6.35% ± 2.38% |
| After T2_gsm8k | 9.04% ± 4.39% | 3.37% ± 2.00% | 6.76% ± 2.99% |
| After T3_sst2 | 10.32% ± 3.71% | 9.81% ± 3.88% | 12.08% ± 4.32% |
| After T4_mbpp | 10.99% ± 7.44% | 2.05% ± 0.48% | 7.69% ± 2.74% |
| After T5_xsum | 12.40% ± 11.08% | 2.88% ± 2.09% | 9.13% ± 5.32% |
| After T6_sciq | 18.01% ± 10.66% | 6.03% ± 4.84% | 13.27% ± 7.73% |
| After T7_multiwoz | 12.08% ± 4.61% | 3.21% ± 1.99% | 8.08% ± 4.57% |

## Method: safety_forever_v2_layer_reg

_Seeds aggregated: 6_

| After Training | ASR (Llama Guard) | ASR (Regex) | ASR (Regex + However harmful override) |
| --- | --- | --- | --- |
| After T1_safety | 12.28% ± 2.70% | 2.44% ± 1.26% | 5.80% ± 1.30% |
| After T2_gsm8k | 9.36% ± 3.81% | 4.33% ± 1.60% | 6.70% ± 1.26% |
| After T3_sst2 | 7.21% ± 4.65% | 5.35% ± 2.67% | 8.11% ± 4.69% |
| After T4_mbpp | 8.81% ± 3.58% | 5.16% ± 3.02% | 9.07% ± 5.30% |
| After T5_xsum | 11.28% ± 5.44% | 3.85% ± 2.23% | 8.91% ± 4.16% |
| After T6_sciq | 11.76% ± 8.45% | 5.42% ± 2.34% | 10.61% ± 3.79% |
| After T7_multiwoz | 12.34% ± 8.52% | 4.58% ± 1.35% | 8.53% ± 3.83% |
