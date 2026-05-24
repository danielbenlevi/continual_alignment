#!/bin/bash
# Smoke test: verify all 3 bug fixes on a 2-GPU run with Qwen3-0.6B (fast).
# Tests: token_type_ids (NOT triggered on Qwen3, but code must not break),
#        gradient_checkpointing (active), dist.barrier() (active in DDP).
# Runs just 1 task (safety only, 1 epoch, 50 steps) for each of 3 methods.
set -euo pipefail

REPO=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/continual_alignment
CONDA_ENV=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/conda-envs/safety_clora_cuda_pip
HF_HOME=/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/hf_cache
OUT=/tmp/smoke_test_fixes_$(date +%s)

submit() {
  local name=$1
  local script=$2
  local extra=$3
  sbatch \
    --account=edu --partition=short \
    --job-name="smoke_${name}" \
    --output="${REPO}/logs/smoke_${name}_%j.out" \
    --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --mem=48G --gres=gpu:2 --time=00:30:00 \
    --exclude=ins082,ins087,ins092 \
    --parsable \
    --wrap="
source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
export HF_HOME=${HF_HOME}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export MASTER_ADDR=localhost
export MASTER_PORT=\$((22355 + SLURM_JOB_ID % 5000))
cd ${REPO}
torchrun --nproc_per_node=2 --master_port=\${MASTER_PORT} \
  ${script} \
  --base_model=Qwen/Qwen3-0.6B \
  --chat_template_mode=never \
  --align_n=200 --alignment_source=wildjailbreak_chat \
  --num_epochs=1 --batch_size=4 \
  --gsm8k_train_n=50 --sst2_train_n=0 --mbpp_train_n=0 \
  --xsum_train_n=0 --sciq_train_n=0 --samsum_train_n=0 \
  --gsm8k_test_n=20 --sst2_test_n=0 --mbpp_test_n=0 \
  --xsum_test_n=0 --sciq_test_n=0 --samsum_test_n=0 \
  --performance_tasks=gsm8k \
  --seed=42 --output_path=${OUT}/${name} ${extra}
echo 'SMOKE_TEST_DONE'
"
}

jid_forever=$(submit "forever" "scripts_training/finetune_forever.py" "--learning_rate=3e-4")
jid_clora=$(submit "clora" "scripts_training/finetune_clora.py" "--learning_rate=1e-3 --clora_lambda=1.0 --clora_k=256")

echo "Submitted smoke tests:"
echo "  FOREVER: job ${jid_forever}  (tests gradient_checkpointing + DDP barrier)"
echo "  CLoRA:   job ${jid_clora}   (tests DDP barrier)"
echo "Check logs: ${REPO}/logs/smoke_*"
echo "All should print SMOKE_TEST_DONE at end if successful."
