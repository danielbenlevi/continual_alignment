#!/bin/bash
# Submit Qwen3-4B-Base + Gemma-3-4b-pt: all 10 methods × 3 orderings × seed 2.
# 2 GPUs each, short partition. Run AFTER models are downloaded to HF cache.
# 10 methods × 3 orderings × 2 models = 60 jobs.
set -euo pipefail

REPO=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/continual_alignment
CONDA_ENV=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/conda-envs/safety_clora_cuda_pip
HF_HOME=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/hf_cache
RUN_ROOT=${REPO}/runs/4b_all_methods_seed2_$(date +%Y%m%d)

SEED=2

declare -A ORDERINGS
ORDERINGS[0]="gsm8k,sst2,mbpp,xsum,sciq,samsum"
ORDERINGS[1]="mbpp,xsum,sst2,sciq,samsum,gsm8k"
ORDERINGS[2]="sciq,gsm8k,samsum,xsum,mbpp,sst2"

submit_job() {
  local model=$1
  local alias=$2
  local method=$3
  local script=$4
  local order_idx=$5
  local extra_args=$6

  local perf_tasks="${ORDERINGS[$order_idx]}"
  local order_tag="order_${order_idx}"
  local out_dir="${RUN_ROOT}/artifacts/${method}/${alias}/seed_${SEED}/${order_tag}"
  local results_json="${RUN_ROOT}/results/${method}/${alias}/seed_${SEED}/${order_tag}.json"
  mkdir -p "$(dirname "$results_json")"

  local job_name="${method:0:7}_${alias:0:4}_o${order_idx}"

  jid=$(sbatch \
    --account=edu \
    --partition=short \
    --job-name="$job_name" \
    --output="${REPO}/logs/4b_${method}_${alias}_s${SEED}_${order_tag}_%j.out" \
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
  --performance_tasks=${perf_tasks} \
  --output_path=${out_dir} \
  --results_json=${results_json} \
  ${extra_args}
")
  echo "Submitted ${method} ${alias} seed=${SEED} order=${order_idx}: job ${jid}"
}

run_all_methods() {
  local model=$1
  local alias=$2
  local order_idx=$3

  submit_job "$model" "$alias" "forever_base" "scripts_training/finetune_forever.py" $order_idx "--learning_rate=3e-4"
  submit_job "$model" "$alias" "safety_forever_base" "scripts_training/finetune_safety_forever.py" $order_idx "--learning_rate=3e-4"
  submit_job "$model" "$alias" "safety_forever_v2_kl" "scripts_training/finetune_safety_forever.py" $order_idx "--learning_rate=3e-4 --enable_safety_token_kl=True --use_safety_reference_model=True"
  submit_job "$model" "$alias" "safety_forever_v2_layer_reg" "scripts_training/finetune_safety_forever.py" $order_idx "--learning_rate=3e-4 --enable_safety_layer_reg_boost=True --use_safety_reference_model=True"
  submit_job "$model" "$alias" "ewcdr_base" "scripts_training/finetune_ewcdr.py" $order_idx "--learning_rate=1e-4 --lamda=30000.0 --omegamax=1e-4"
  submit_job "$model" "$alias" "ewcdr_safety" "scripts_training/finetune_safety_ewcdr.py" $order_idx "--learning_rate=1e-4 --lamda=30000.0 --omegamax=1e-4"
  submit_job "$model" "$alias" "clora_random" "scripts_training/finetune_clora.py" $order_idx "--learning_rate=1e-3 --clora_lambda=1.0 --clora_k=256"
  submit_job "$model" "$alias" "clora_safety" "scripts_training/finetune_safety_clora.py" $order_idx "--learning_rate=1e-3 --clora_lambda=1.0 --clora_k=256"
  submit_job "$model" "$alias" "olora_standard" "scripts_training/finetune_olora.py" $order_idx "--learning_rate=1e-3 --olora_lambda_1=0.5"
  submit_job "$model" "$alias" "olora_safety" "scripts_training/finetune_safety_olora.py" $order_idx "--learning_rate=1e-3 --olora_lambda_1=0.5 --olora_safety_lambda_1=2.5"
}

echo "=== Submitting 4B models all methods seed=${SEED} ==="
echo "Run root: ${RUN_ROOT}"
echo "Models: Qwen3-4B-Base + gemma-3-4b-pt"
echo ""

for order_idx in 0 1 2; do
  echo "--- Order ${order_idx} ---"
  run_all_methods "Qwen/Qwen3-4B-Base" "Qwen3-4B-Base" $order_idx
  run_all_methods "google/gemma-3-4b-pt" "gemma-3-4b-pt" $order_idx
done

echo ""
echo "=== Done. Results in ${RUN_ROOT}/results/ ==="
