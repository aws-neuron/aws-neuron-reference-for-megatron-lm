DATA_PATH=~/examples_datasets/gpt2/my-gpt2_text_document
CHECKPOINT_PATH=~/examples

export MASTER_ADDR='localhost'
export MASTER_PORT=6000
export NEURON_NUM_DEVICES=32

export OMP_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export XLA_IO_THREAD_POOL_SIZE=2
export XLA_THREAD_POOL_SIZE=2
export TF_GRPC_WORKER_CACHE_QUEUES=4
export TF_GRPC_WORKER_CACHE_THREADS=4

export NEURON_INTERNAL_TRANSFER_ALL_PARAMETERS_WITH_STATIC_RING=1

python3 pretrain_gpt_mp.py \
    --tensor-model-parallel-size 32 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters 10000 \
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
    --eval-interval 1000 \
    --eval-iters 10 \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --no-contiguous-buffers-in-local-ddp \
    --tensorboard-dir ./tb_gpt2_32layer_fp32 \
    |& tee run_log_gpt2_32layer_fp32
