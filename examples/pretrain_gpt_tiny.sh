#! /bin/bash

# Runs the "345M" parameter model

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR='localhost'
export MASTER_PORT=6000

DATA_PATH=examples_new/example_text_document
CHECKPOINT_PATH=examples_new

#export NEURON_EXTRACT_GRAPHS_ONLY=1
#export NEURON_FALL_BACK_TO_NULL_NEFF=1

python3.6 pretrain_gpt.py "$@" \
       --num-layers 2 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 1 \
       --global-batch-size 1\
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 1000 \
       --no-masked-softmax-fusion \
       --no-bias-gelu-fusion \
       --no-bias-dropout-fusion \
       --lr-decay-iters 320000 \
       --data-path $DATA_PATH \
       --vocab-file examples_new/gpt2-vocab.json \
       --merge-file examples_new/gpt2-merges.txt \
       --data-impl mmap \
       --split 100,0,0 \
       --distributed-backend gloo \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --no-contiguous-buffers-in-local-ddp \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --tensorboard-dir examples_new/tb
#       --fp16
