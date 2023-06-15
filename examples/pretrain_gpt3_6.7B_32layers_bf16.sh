#! /bin/bash
set -o pipefail

MODEL_CONFIG_NAME=gpt3_6.7B_32layers_bf16

DATA_PATH=~/examples_datasets/gpt2/my-gpt2_text_document
CHECKPOINT_PATH=chkpt_${MODEL_CONFIG_NAME}

NUM_NEURONCORES=32
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"

# Keep only 3 number of graphs loaded in Neuron runtime for each process to reduce device mem usage
export NEURON_NUM_RECENT_MODELS_TO_KEEP=3
# Mark all parameter transfers as static to enable runtime optimizations for wrapped torch.nn modules
export NEURON_TRANSFER_ALL_PARAMETERS_WITH_STATIC_RING=1
# Enables custom lowering for Softmax operation to enable compiler optimizations and improve GPT performance
export NEURON_FUSE_SOFTMAX=1
# Cast training to BF16 and enable stochastic rounding
export XLA_USE_BF16=1

# Workaround "Too many open files" error with GPT training on U20 server AMI
ulimit -n 8192

TRAIN_ITERS=10000
TB_DIR=./tb_${MODEL_CONFIG_NAME}
if [[ "$NEURON_EXTRACT_GRAPHS_ONLY" == "1" ]]; then
    TRAIN_ITERS=65
    TB_DIR=/tmp/parallel_compile_ignored_tb_output
fi

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters 320000 \
    --data-path $DATA_PATH \
    --vocab-file ~/examples_datasets/gpt2/gpt2-vocab.json \
    --merge-file ~/examples_datasets/gpt2/gpt2-merges.txt \
    --data-impl mmap \
    --split 100,0,0 \
    --distributed-backend xla \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1 \
    --lr-warmup-fraction .01 \
    --log-interval 1 \
    --tensorboard-log-interval 1 \
    --eval-interval $TRAIN_ITERS \
    --eval-iters 1000 \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --no-contiguous-buffers-in-local-ddp \
    --save-xser $CHECKPOINT_PATH \
    --save-interval 2000 \
    --keep-last-checkpoint-only \
    --use-cpu-initialization \
    --tensorboard-dir $TB_DIR \
    |& tee run_log_$MODEL_CONFIG_NAME.txt &
wait %1

ret_val=$?
if [ $ret_val -eq 0 ]; then
    success=1
else
    success=0
fi

dump_to_s3_update_json_scr=../../dump_to_s3_update_test_json.sh
if [ -e $dump_to_s3_update_json_scr ]; then
    $dump_to_s3_update_json_scr $@ --key=inference_success --value=$success || echo "Unable to update test result JSON."
else
    echo "WARNING: Script $dump_to_s3_update_json_scr not found. Not updating test result JSON."
fi

exit $ret_val
