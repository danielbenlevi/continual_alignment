#!/bin/bash
# Submit LoRA baseline for Qwen3-4B-Base and Gemma-3-4b-pt.
# 2 GPUs each, short partition. Run AFTER models are downloaded to HF cache.
# 3 orderings × 3 seeds × 2 models = 18 jobs.
set -euo pipefail

REPO=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/continual_alignment
CONDA_ENV=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/conda-envs/safety_clora_cuda_pip
HF_HOME=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/hf_cache
RUN_ROOT=${REPO}/runs/lora_baseline_4b_$(date +%Y%m%d)

SEEDS=(0 1 2)

declare -A ORDERINGS
ORDERINGS[0]="gsm8k,sst2,mbpp,xsum,sciq,samsum"
ORDERINGS[1]="mbpp,xsum,sst2,sciq,samsum,gsm8k"
ORDERINGS[2]="sciq,gsm8k,samsum,xsum,mbpp,sst2"

submit_job() {
  local model=$1
  local alias=$2
  local order_idx=$3
  local seed=$4

  local perf_tasks="${ORDERINGS[$order_idx]}"
  local order_tag="order_${order_idx}"
  local out_dir="${RUN_ROOT}/artifacts/lora_baseline/${alias}/seed_${seed}/${order_tag}"
  local results_json="${RUN_ROOT}/results/lora_baseline/${alias}/seed_${seed}/${order_tag}.json"
  mkdir -p "$(dirname "$results_json")"

  local job_name="lora_${alias:0:6}_s${seed}_o${order_idx}"

  jid=$(sbatch \
    --account=edu \
    --partition=short \
    --job-name="$job_name" \
    --output="${REPO}/logs/lora4b_${alias}_seed${seed}_${order_tag}_%j.out" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --mem=64G --gres=gpu:2 \
    --time=12:00:00 \
    --exclude=ins082,ins087,ins092 \
    --parsable \
    --wrap="
source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export HF_HOME=${HF_HOME}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=localhost
export MASTER_PORT=\$((12355 + SLURM_JOB_ID % 10000))
cd ${REPO}
torchrun --nproc_per_node=2 --master_addr=localhost --master_port=\$((12355 + SLURM_JOB_ID % 10000)) \
  scripts_training/finetune_lora.py \
  --base_model=${model} \
  --chat_template_mode=never \
  --align_n=10000 --alignment_source=wildjailbreak_chat \
  --num_epochs=3 --lora_r=8 --lora_alpha=16 --lora_dropout=0.05 \
  --batch_size=8 --eval_batch_size=64 \
  --learning_rate=3e-4 \
  --gsm8k_train_n=2000 --sst2_train_n=2000 --mbpp_train_n=-1 \
  --xsum_train_n=2000 --sciq_train_n=2000 --samsum_train_n=2000 \
  --gsm8k_test_n=500 --sst2_test_n=500 --mbpp_test_n=500 \
  --xsum_test_n=500 --sciq_test_n=500 --samsum_test_n=500 \
  --seed=${seed} \
  --performance_tasks=${perf_tasks} \
  --output_path=${out_dir} \
  --results_json=${results_json}
")
  echo "Submitted lora_baseline ${alias} seed=${seed} order=${order_idx}: job ${jid}"
}

echo "=== Submitting LoRA baseline for 4B models ==="
echo "Run root: ${RUN_ROOT}"
echo ""

for seed in "${SEEDS[@]}"; do
  for order_idx in 0 1 2; do
    submit_job "Qwen/Qwen3-4B-Base" "Qwen3-4B-Base" $order_idx $seed
    submit_job "google/gemma-3-4b-pt" "gemma-3-4b-pt" $order_idx $seed
  done
done

echo ""
echo "=== Done. Results in ${RUN_ROOT}/results/ ==="
