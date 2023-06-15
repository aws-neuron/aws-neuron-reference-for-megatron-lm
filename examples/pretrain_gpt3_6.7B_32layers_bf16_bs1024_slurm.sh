#! /bin/bash
set -o pipefail

MODEL_CONFIG_NAME=gpt3_6.7B_32layers_bf16
 
# Enable Elastic Fabric Adapter for higher networking performance
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

DATA_PATH=~/examples_datasets/gpt2/my-gpt2_text_document
 
MASTER_ADDR=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
MASTER_PORT=2022
NUM_NEURONCORES=32
 
WORLD_SIZE_JOB=$SLURM_NTASKS
RANK_NODE=$SLURM_NODEID
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES --nnodes $WORLD_SIZE_JOB --node_rank $RANK_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS
 
CHECKPOINT_PATH=chkpt_${MODEL_CONFIG_NAME}_${WORLD_SIZE_JOB}

# Keep only 3 number of graphs loaded in Neuron runtime for each process to reduce device mem usage
export NEURON_NUM_RECENT_MODELS_TO_KEEP=3
# Mark all parameter transfers as static to enable runtime optimizations for wrapped torch.nn modules
export NEURON_TRANSFER_ALL_PARAMETERS_WITH_STATIC_RING=1
# Enables custom lowering for Softmax operation to enable compiler optimizations and improve GPT performance
export NEURON_FUSE_SOFTMAX=1
# Cast training to BF16 and enable stochastic rounding
export XLA_USE_BF16=1
# Increase Neuron RT execution timeout in case slow compilation causes Neuron RT to wait longer than default timeout
export NEURON_RT_EXEC_TIMEOUT=600

# Separate NeuronCache dir per node, workaround limitation to file locking on NFS
export NEURON_CC_FLAGS="--cache_dir=$HOME/neuron_cache/gpt/`hostname`"
 
TRAIN_ITERS=143051
TB_DIR=./tb_${MODEL_CONFIG_NAME}
# Run fewer steps and ignore tb output when extract graphs only (neuron_parallel_compile)
if [[ "$NEURON_EXTRACT_GRAPHS_ONLY" == "1" ]]; then
    # Using larger trial count to workaround extra recompilation due to https://github.com/pytorch/xla/issues/4994
    TRAIN_ITERS=325
    TB_DIR=/tmp/parallel_compile_ignored_tb_output
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
    --save-xser $CHECKPOINT_PATH \
    --save-interval 2000 \
    --keep-last-checkpoint-only \
    --use-cpu-initialization \
    --tensorboard-dir $TB_DIR \
    |& tee run_log_$MODEL_CONFIG_NAME.$RANK_NODE.$WORLD_SIZE_JOB.txt &
wait %1
 
ret_val=$?

if [ $ret_val -eq 0 ] ; then
    msg="SUCCESS"
elif [ $ret_val -eq 2 ] ; then
    msg="SCANCEL/INTERRUPT"
else
    msg="INTERNAL FAILURE"
    # Uncomment lines below to requeue after internal failure (make sure the script doesn't fail)
    #msg="INTERNAL FAILURE - HARDWARE ISSUE? Requeue JOB ID ${SLURM_JOB_ID} - use scancel to terminate"
    #scontrol requeue ${SLURM_JOB_ID}
fi
echo $msg

if [ $ret_val -eq 0 ]; then
    success=1
else
    success=0
fi

# Below is for testing only, not needed for actual execution
dump_to_s3_update_json_scr=../../dump_to_s3_update_test_json.sh
if [ -e $dump_to_s3_update_json_scr ]; then
    $dump_to_s3_update_json_scr $@ --key=inference_success --value=$success || echo "Unable to update test result JSON."
else
    echo "WARNING: Script $dump_to_s3_update_json_scr not found. Not updating test result JSON."
fi

exit $ret_val
