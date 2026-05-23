#!/bin/bash
# Submit Qwen3-0.6B-Base all 10 methods × 3 orderings × seeds 0,1,2 (90 jobs).
# 1 GPU each, short partition. Model already in HF cache.
set -euo pipefail

REPO=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/continual_alignment
CONDA_ENV=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/conda-envs/safety_clora_cuda_pip
HF_HOME=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/hf_cache
RUN_ROOT=${REPO}/runs/qwen06b_all_methods_$(date +%Y%m%d)

BASE_MODEL="Qwen/Qwen3-0.6B-Base"
ALIAS="Qwen3-0.6B-Base"
SEEDS=(0 1 2)

declare -A ORDERINGS
ORDERINGS[0]="gsm8k,sst2,mbpp,xsum,sciq,samsum"
ORDERINGS[1]="mbpp,xsum,sst2,sciq,samsum,gsm8k"
ORDERINGS[2]="sciq,gsm8k,samsum,xsum,mbpp,sst2"

COMMON_ARGS="--base_model=${BASE_MODEL} --chat_template_mode=never \
  --align_n=10000 --alignment_source=wildjailbreak_chat \
  --num_epochs=3 --lora_r=8 --lora_alpha=16 --lora_dropout=0.05 \
  --batch_size=8 --eval_batch_size=64 \
  --gsm8k_train_n=2000 --sst2_train_n=2000 --mbpp_train_n=-1 \
  --xsum_train_n=2000 --sciq_train_n=2000 --samsum_train_n=2000 \
  --gsm8k_test_n=500 --sst2_test_n=500 --mbpp_test_n=500 \
  --xsum_test_n=500 --sciq_test_n=500 --samsum_test_n=500"

submit_job() {
  local method=$1
  local script=$2
  local order_idx=$3
  local seed=$4
  local extra_args=$5

  local perf_tasks="${ORDERINGS[$order_idx]}"
  local order_tag="order_${order_idx}"
  local out_dir="${RUN_ROOT}/artifacts/${method}/${ALIAS}/seed_${seed}/${order_tag}"
  local results_json="${RUN_ROOT}/results/${method}/${ALIAS}/seed_${seed}/${order_tag}.json"
  mkdir -p "$(dirname "$results_json")"

  local job_name="${method:0:8}_q06_s${seed}_o${order_idx}"

  jid=$(sbatch \
    --account=edu \
    --partition=short \
    --job-name="$job_name" \
    --output="${REPO}/logs/qwen06b_${method}_seed${seed}_${order_tag}_%j.out" \
    --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --mem=40G --gres=gpu:1 \
    --time=12:00:00 \
    --exclude=ins082 \
    --parsable \
    --wrap="
source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export HF_HOME=${HF_HOME}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd ${REPO}
python ${script} \
  ${COMMON_ARGS} \
  --seed=${seed} \
  --performance_tasks=${perf_tasks} \
  --output_path=${out_dir} \
  --results_json=${results_json} \
  ${extra_args}
")
  echo "Submitted ${method} seed=${seed} order=${order_idx}: job ${jid}"
}

echo "=== Submitting Qwen3-0.6B-Base all methods ==="
echo "Run root: ${RUN_ROOT}"
echo ""

for seed in "${SEEDS[@]}"; do
  for order_idx in 0 1 2; do
    # FOREVER methods
    submit_job "forever_base" "scripts_training/finetune_forever.py" $order_idx $seed "--learning_rate=3e-4"
    submit_job "safety_forever_base" "scripts_training/finetune_safety_forever.py" $order_idx $seed "--learning_rate=3e-4"
    submit_job "safety_forever_v2_kl" "scripts_training/finetune_safety_forever.py" $order_idx $seed "--learning_rate=3e-4 --enable_safety_token_kl=True --use_safety_reference_model=True"
    submit_job "safety_forever_v2_layer_reg" "scripts_training/finetune_safety_forever.py" $order_idx $seed "--learning_rate=3e-4 --enable_safety_layer_reg_boost=True --use_safety_reference_model=True"

    # EWC-DR methods
    submit_job "ewcdr_base" "scripts_training/finetune_ewcdr.py" $order_idx $seed "--learning_rate=1e-4 --lamda=30000.0 --omegamax=1e-4"
    submit_job "ewcdr_safety" "scripts_training/finetune_safety_ewcdr.py" $order_idx $seed "--learning_rate=1e-4 --lamda=30000.0 --omegamax=1e-4"

    # CLoRA methods
    submit_job "clora_random" "scripts_training/finetune_clora.py" $order_idx $seed "--learning_rate=1e-3 --clora_lambda=1.0 --clora_k=256"
    submit_job "clora_safety" "scripts_training/finetune_safety_clora.py" $order_idx $seed "--learning_rate=1e-3 --clora_lambda=1.0 --clora_k=256"

    # O-LoRA methods
    submit_job "olora_standard" "scripts_training/finetune_olora.py" $order_idx $seed "--learning_rate=1e-3 --olora_lambda_1=0.5"
    submit_job "olora_safety" "scripts_training/finetune_safety_olora.py" $order_idx $seed "--learning_rate=1e-3 --olora_lambda_1=0.5 --olora_safety_lambda_1=2.5"
  done
done

echo ""
echo "=== Done submitting. Check results in ${RUN_ROOT}/results/ ==="
