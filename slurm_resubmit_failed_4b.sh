#!/bin/bash
# Resubmit only the MISSING 4B jobs after fixing:
#   - Gemma-3-4b-pt: DDP vision-tower bug (fixed in trainer.py)
#   - Qwen3-4B-Base: NCCL timeout (transient, retry) + gradient checkpointing fix
# Skips jobs where the results JSON already exists.
# Run from within continual_alignment/
set -euo pipefail

REPO=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/continual_alignment
CONDA_ENV=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/conda-envs/safety_clora_cuda_pip
HF_HOME=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/hf_cache

# Use same dated run roots so results land in the right place
SEED2_ROOT=${REPO}/runs/4b_all_methods_seed2_20260523
LORA_ROOT=${REPO}/runs/lora_baseline_4b_20260523
SEED=2

declare -A ORDERINGS
ORDERINGS[0]="gsm8k,sst2,mbpp,xsum,sciq,samsum"
ORDERINGS[1]="mbpp,xsum,sst2,sciq,samsum,gsm8k"
ORDERINGS[2]="sciq,gsm8k,samsum,xsum,mbpp,sst2"

submit_if_missing() {
  local results_json=$1
  local script=$2
  local model=$3
  local alias=$4
  local out_dir=$5
  local job_name=$6
  local extra_args=$7

  if [[ -f "$results_json" ]]; then
    echo "SKIP (exists): $results_json"
    return
  fi

  mkdir -p "$(dirname "$results_json")"
  jid=$(sbatch \
    --account=edu \
    --partition=short \
    --job-name="$job_name" \
    --output="${REPO}/logs/resubmit_${job_name}_%j.out" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --mem=64G --gres=gpu:2 \
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
export MASTER_ADDR=localhost
export MASTER_PORT=\$((12355 + SLURM_JOB_ID % 10000))
cd ${REPO}
torchrun --nproc_per_node=2 --master_addr=localhost --master_port=\$((12355 + SLURM_JOB_ID % 10000)) \
  ${script} \
  --base_model=${model} \
  --chat_template_mode=never \
  --align_n=10000 --alignment_source=wildjailbreak_chat \
  --num_epochs=3 --lora_r=8 --lora_alpha=16 --lora_dropout=0.05 \
  --batch_size=8 --eval_batch_size=64 \
  --gsm8k_train_n=2000 --sst2_train_n=2000 --mbpp_train_n=-1 \
  --xsum_train_n=2000 --sciq_train_n=2000 --samsum_train_n=2000 \
  --gsm8k_test_n=500 --sst2_test_n=500 --mbpp_test_n=500 \
  --xsum_test_n=500 --sciq_test_n=500 --samsum_test_n=500 \
  --seed=\${SEED_ARG} \
  --performance_tasks=\${PERF_TASKS_ARG} \
  --output_path=${out_dir} \
  --results_json=${results_json} \
  ${extra_args}
"
)
  echo "Submitted: $results_json → job $jid"
}

# This approach is cleaner — inline variables in the wrap:
submit_if_missing_inline() {
  local results_json=$1
  local script=$2
  local model=$3
  local alias=$4
  local out_dir=$5
  local job_name=$6
  local seed=$7
  local perf_tasks=$8
  local extra_args=$9

  if [[ -f "$results_json" ]]; then
    echo "SKIP (exists): $results_json"
    return
  fi

  mkdir -p "$(dirname "$results_json")"
  jid=$(sbatch \
    --account=edu \
    --partition=short \
    --job-name="$job_name" \
    --output="${REPO}/logs/resubmit_${job_name}_%j.out" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --mem=64G --gres=gpu:2 \
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
export MASTER_ADDR=localhost
export MASTER_PORT=\$((12355 + SLURM_JOB_ID % 10000))
cd ${REPO}
torchrun --nproc_per_node=2 --master_addr=localhost --master_port=\$((12355 + SLURM_JOB_ID % 10000)) \
  ${script} \
  --base_model=${model} \
  --chat_template_mode=never \
  --align_n=10000 --alignment_source=wildjailbreak_chat \
  --num_epochs=3 --lora_r=8 --lora_alpha=16 --lora_dropout=0.05 \
  --batch_size=8 --eval_batch_size=64 \
  --gsm8k_train_n=2000 --sst2_train_n=2000 --mbpp_train_n=-1 \
  --xsum_train_n=2000 --sciq_train_n=2000 --samsum_train_n=2000 \
  --gsm8k_test_n=500 --sst2_test_n=500 --mbpp_test_n=500 \
  --xsum_test_n=500 --sciq_test_n=500 --samsum_test_n=500 \
  --seed=${seed} \
  --performance_tasks=${perf_tasks} \
  --output_path=${out_dir} \
  --results_json=${results_json} \
  ${extra_args}
")
  echo "Submitted: $(basename $results_json) (seed=${seed} order=$(echo $perf_tasks | cut -d, -f1)) → job $jid"
}

# ============================================================
# Part 1: LoRA baseline - missing jobs
# ============================================================
echo "=== LoRA Baseline: Submitting missing jobs ==="

# Missing Qwen3-4B-Base: seed_1/order_0, seed_1/order_1, seed_2/order_2
for seed in 1; do
  for order_idx in 0 1; do
    perf="${ORDERINGS[$order_idx]}"
    out="${LORA_ROOT}/artifacts/lora_baseline/Qwen3-4B-Base/seed_${seed}/order_${order_idx}"
    res="${LORA_ROOT}/results/lora_baseline/Qwen3-4B-Base/seed_${seed}/order_${order_idx}.json"
    jname="lora_q4b_s${seed}_o${order_idx}"
    submit_if_missing_inline "$res" "scripts_training/finetune_lora.py" "Qwen/Qwen3-4B-Base" "Qwen3-4B-Base" "$out" "$jname" "$seed" "$perf" ""
  done
done

# seed_2/order_2
seed=2; order_idx=2
perf="${ORDERINGS[$order_idx]}"
out="${LORA_ROOT}/artifacts/lora_baseline/Qwen3-4B-Base/seed_${seed}/order_${order_idx}"
res="${LORA_ROOT}/results/lora_baseline/Qwen3-4B-Base/seed_${seed}/order_${order_idx}.json"
submit_if_missing_inline "$res" "scripts_training/finetune_lora.py" "Qwen/Qwen3-4B-Base" "Qwen3-4B-Base" "$out" "lora_q4b_s2_o2" "$seed" "$perf" ""

# All Gemma-3-4b-pt LoRA baseline jobs (9 total)
for seed in 0 1 2; do
  for order_idx in 0 1 2; do
    perf="${ORDERINGS[$order_idx]}"
    out="${LORA_ROOT}/artifacts/lora_baseline/gemma-3-4b-pt/seed_${seed}/order_${order_idx}"
    res="${LORA_ROOT}/results/lora_baseline/gemma-3-4b-pt/seed_${seed}/order_${order_idx}.json"
    jname="lora_gem_s${seed}_o${order_idx}"
    submit_if_missing_inline "$res" "scripts_training/finetune_lora.py" "google/gemma-3-4b-pt" "gemma-3-4b-pt" "$out" "$jname" "$seed" "$perf" ""
  done
done

# ============================================================
# Part 2: All methods seed 2 - missing jobs
# ============================================================
echo ""
echo "=== All Methods Seed 2: Submitting missing jobs ==="

declare -A METHODS
METHODS[forever_base]="scripts_training/finetune_forever.py --learning_rate=3e-4"
METHODS[safety_forever_base]="scripts_training/finetune_safety_forever.py --learning_rate=3e-4"
METHODS[safety_forever_v2_kl]="scripts_training/finetune_safety_forever.py --learning_rate=3e-4 --enable_safety_token_kl=True --use_safety_reference_model=True"
METHODS[safety_forever_v2_layer_reg]="scripts_training/finetune_safety_forever.py --learning_rate=3e-4 --enable_safety_layer_reg_boost=True --use_safety_reference_model=True"
METHODS[ewcdr_base]="scripts_training/finetune_ewcdr.py --learning_rate=1e-4 --lamda=30000.0 --omegamax=1e-4"
METHODS[ewcdr_safety]="scripts_training/finetune_safety_ewcdr.py --learning_rate=1e-4 --lamda=30000.0 --omegamax=1e-4"
METHODS[clora_random]="scripts_training/_clora_olora_common.py --learning_rate=1e-3 --clora_lambda=1.0 --clora_k=256"
METHODS[clora_safety]="scripts_training/_clora_olora_common.py --learning_rate=1e-3 --clora_lambda=1.0 --clora_k=256"
METHODS[olora_standard]="scripts_training/_clora_olora_common.py --learning_rate=1e-3 --olora_lambda_1=0.5"
METHODS[olora_safety]="scripts_training/_clora_olora_common.py --learning_rate=1e-3 --olora_lambda_1=0.5 --olora_safety_lambda_1=2.5"

# Actually use the correct individual scripts
submit_method() {
  local method=$1
  local model=$2
  local alias=$3
  local order_idx=$4
  local script=$5
  local extra=$6

  local perf="${ORDERINGS[$order_idx]}"
  local out="${SEED2_ROOT}/artifacts/${method}/${alias}/seed_${SEED}/order_${order_idx}"
  local res="${SEED2_ROOT}/results/${method}/${alias}/seed_${SEED}/order_${order_idx}.json"
  local jname="${method:0:7}_${alias:0:4}_o${order_idx}"

  if [[ -f "$res" ]]; then
    echo "SKIP (exists): ${method}/${alias}/order_${order_idx}"
    return
  fi

  mkdir -p "$(dirname "$res")"
  jid=$(sbatch \
    --account=edu \
    --partition=short \
    --job-name="$jname" \
    --output="${REPO}/logs/resubmit_${method}_${alias}_s${SEED}_order${order_idx}_%j.out" \
    --nodes=1 --ntasks=1 --cpus-per-task=8 \
    --mem=64G --gres=gpu:2 \
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
export MASTER_ADDR=localhost
export MASTER_PORT=\$((12355 + SLURM_JOB_ID % 10000))
cd ${REPO}
torchrun --nproc_per_node=2 --master_addr=localhost --master_port=\$((12355 + SLURM_JOB_ID % 10000)) \
  ${script} \
  --base_model=${model} \
  --chat_template_mode=never \
  --align_n=10000 --alignment_source=wildjailbreak_chat \
  --num_epochs=3 --lora_r=8 --lora_alpha=16 --lora_dropout=0.05 \
  --batch_size=8 --eval_batch_size=64 \
  --gsm8k_train_n=2000 --sst2_train_n=2000 --mbpp_train_n=-1 \
  --xsum_train_n=2000 --sciq_train_n=2000 --samsum_train_n=2000 \
  --gsm8k_test_n=500 --sst2_test_n=500 --mbpp_test_n=500 \
  --xsum_test_n=500 --sciq_test_n=500 --samsum_test_n=500 \
  --seed=${SEED} \
  --performance_tasks=${perf} \
  --output_path=${out} \
  --results_json=${res} \
  ${extra}
")
  echo "Submitted: ${method}/${alias}/seed_${SEED}/order_${order_idx} → job $jid"
}

for order_idx in 0 1 2; do
  for model_pair in "Qwen/Qwen3-4B-Base:Qwen3-4B-Base" "google/gemma-3-4b-pt:gemma-3-4b-pt"; do
    model="${model_pair%%:*}"
    alias="${model_pair##*:}"

    submit_method "forever_base" "$model" "$alias" $order_idx "scripts_training/finetune_forever.py" "--learning_rate=3e-4"
    submit_method "safety_forever_base" "$model" "$alias" $order_idx "scripts_training/finetune_safety_forever.py" "--learning_rate=3e-4"
    submit_method "safety_forever_v2_kl" "$model" "$alias" $order_idx "scripts_training/finetune_safety_forever.py" "--learning_rate=3e-4 --enable_safety_token_kl=True --use_safety_reference_model=True"
    submit_method "safety_forever_v2_layer_reg" "$model" "$alias" $order_idx "scripts_training/finetune_safety_forever.py" "--learning_rate=3e-4 --enable_safety_layer_reg_boost=True --use_safety_reference_model=True"
    submit_method "ewcdr_base" "$model" "$alias" $order_idx "scripts_training/finetune_ewcdr.py" "--learning_rate=1e-4 --lamda=30000.0 --omegamax=1e-4"
    submit_method "ewcdr_safety" "$model" "$alias" $order_idx "scripts_training/finetune_safety_ewcdr.py" "--learning_rate=1e-4 --lamda=30000.0 --omegamax=1e-4"
    submit_method "clora_random" "$model" "$alias" $order_idx "scripts_training/finetune_clora.py" "--learning_rate=1e-3 --clora_lambda=1.0 --clora_k=256"
    submit_method "clora_safety" "$model" "$alias" $order_idx "scripts_training/finetune_safety_clora.py" "--learning_rate=1e-3 --clora_lambda=1.0 --clora_k=256"
    submit_method "olora_standard" "$model" "$alias" $order_idx "scripts_training/finetune_olora.py" "--learning_rate=1e-3 --olora_lambda_1=0.5"
    submit_method "olora_safety" "$model" "$alias" $order_idx "scripts_training/finetune_safety_olora.py" "--learning_rate=1e-3 --olora_lambda_1=0.5 --olora_safety_lambda_1=2.5"
  done
done

echo ""
echo "=== Done submitting. Check results in:"
echo "  ${SEED2_ROOT}/results/"
echo "  ${LORA_ROOT}/results/"
