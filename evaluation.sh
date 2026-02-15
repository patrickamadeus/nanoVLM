export HF_HOME=~/users/patrick/huggingface
: "${HF_TOKEN:?Set HF_TOKEN in environment before running evaluation.sh}"
export HF_TOKEN
export CUDA_VISIBLE_DEVICES=0
LIMIT=100
TASKS=mmstar
BATCH_SIZE=16
CP=patrickamadeus/momh-2k1img-step-1000


python evaluation.py \
    --tasks ${TASKS} \
    --limit ${LIMIT} \
    --batch_size ${BATCH_SIZE} \
    --device cuda \
    --output_path eval_results/$(basename ${CP}) \
    --model ${CP} \
    --write_out \
    --log_samples \
    --mode nanovlm
