#! /bin/bash

# Runs the "345M" parameter model


DATA_PATH=~/examples_datasets/gpt2/my-gpt2_text_document
CHECKPOINT_PATH=~/examples
#export NEURON_EXTRACT_GRAPHS_ONLY=1
#export NEURON_FALL_BACK_TO_NULL_NEFF=1

export MASTER_ADDR='localhost'
export MASTER_PORT=6000
export NEURON_NUM_DEVICES=2

python3 pretrain_gpt_mp.py  \
       --tensor-model-parallel-size 2 \
       --num-layers 2 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --micro-batch-size 1 \
       --global-batch-size 1 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 1000 \
       --no-masked-softmax-fusion \
       --no-bias-gelu-fusion \
       --no-bias-dropout-fusion \
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
       --clip-grad 0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --no-contiguous-buffers-in-local-ddp \
       --no-async-tensor-model-parallel-allreduce \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --tensorboard-dir tb

#       --fp16
