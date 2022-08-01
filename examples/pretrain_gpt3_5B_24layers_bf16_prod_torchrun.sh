#! /bin/bash
set -o pipefail

DATA_PATH=~/examples_datasets/gpt2/my-gpt2_text_document
# Change for multinode config
#DATA_PATH=/shared/my-gpt2_text_document

export MASTER_ADDR='localhost'
# Change for multinode config
#MASTER_ADDR=10.1.200.202
MASTER_PORT=2022
NEURON_NUM_DEVICES=32

NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NEURON_NUM_DEVICES*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $NEURON_NUM_DEVICES --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

export OMP_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export XLA_IO_THREAD_POOL_SIZE=2
export XLA_THREAD_POOL_SIZE=2
export TF_GRPC_WORKER_CACHE_QUEUES=4
export TF_GRPC_WORKER_CACHE_THREADS=4

export NEURON_NUM_RECENT_MODELS_TO_KEEP=3
export NEURON_RT_ONE_TMPBUF_PAGE_SIZE_MB=1024
export NEURON_INTERNAL_TRANSFER_ALL_PARAMETERS_WITH_STATIC_RING=1

export NEURON_RT_STOCHASTIC_ROUNDING_SEED=0
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
export XLA_USE_BF16=1

#This flag will be made default in future
export XLA_TRANSFER_SCALAR_ASYNC=1

#Turning this on results in a hang after tracing, but need it
#to enable softmax lowering
#export NEURON_INTERNAL_FUSE_SOFTMAX=1

TRAIN_ITERS=10000
if [[ "$NEURON_EXTRACT_GRAPHS_ONLY" == "1" ]]; then
    TRAIN_ITERS=125
fi

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --num-layers 24 \
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
    --split 90,10,0 \
    --distributed-backend xla \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1 \
    --lr-warmup-fraction .01 \
    --log-interval 4 \
    --tensorboard-log-interval 4 \
    --eval-interval $TRAIN_ITERS \
    --eval-iters 1000 \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --no-contiguous-buffers-in-local-ddp \
    --tensorboard-dir ./tb_gpt2_24layer_bf16 \
    |& tee run_log_gpt2_24layer_bf16_prod

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
