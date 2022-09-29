#! /bin/bash
set -o pipefail
 
sudo modprobe -r neuron; sudo modprobe neuron
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
 
DATA_PATH=~/examples_datasets/gpt2/my-gpt2_text_document
 
MASTER_ADDR=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
MASTER_PORT=2022
NUM_NEURONCORES=32
 
WORLD_SIZE_JOB=$SLURM_NTASKS
RANK_NODE=$SLURM_NODEID
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES --nnodes $WORLD_SIZE_JOB --node_rank $RANK_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS
 
export NEURON_NUM_RECENT_MODELS_TO_KEEP=3
export NEURON_INTERNAL_TRANSFER_ALL_PARAMETERS_WITH_STATIC_RING=1
 
export NEURON_RT_STOCHASTIC_ROUNDING_SEED=0
 
export XLA_USE_BF16=1
export NEURON_CC_FLAGS="--model-type transformer"
 
 
TRAIN_ITERS=143051
if [[ "$NEURON_EXTRACT_GRAPHS_ONLY" == "1" ]]; then
    TRAIN_ITERS=65
fi
 
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --tensor-model-parallel-size 8 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --micro-batch-size 1 \
    --global-batch-size 1024 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters 123977 \
    --data-path $DATA_PATH \
    --vocab-file ~/examples_datasets/gpt2/gpt2-vocab.json \
    --merge-file ~/examples_datasets/gpt2/gpt2-merges.txt \
    --data-impl mmap \
    --split 100,0,0 \
    --distributed-backend xla \
    --lr 0.00012 \
    --lr-decay-style cosine \
    --min-lr 1.2e-5 \
    --weight-decay 1e-1 \
    --clip-grad 1 \
    --lr-warmup-fraction 0.00125 \
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
    --init-method-std 0.006 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --save 32_layer_$WORLD_SIZE_JOB \
    --save-interval 1500 \
    --use-cpu-initialization \
    --load 32_layer_$WORLD_SIZE_JOB \
    --tensorboard-dir ./tb_gpt3_32layer_bf16 \
    |& tee run_log_gpt3_32layer_bf16_torchrun.$RANK_NODE.$WORLD_SIZE_JOB.log
 

dump_to_s3_update_json_scr=../../dump_to_s3_update_test_json.sh
if [ -e $dump_to_s3_update_json_scr ]; then
    $dump_to_s3_update_json_scr $@ --key=inference_success --value=$success || echo "Unable to update test result JSON."
else
    echo "WARNING: Script $dump_to_s3_update_json_scr not found. Not updating test result JSON."
fi

ret_val=$?
if [ $ret_val -eq 0 ]; then
    success=1
else
    success=0
fi

exit $ret_val
