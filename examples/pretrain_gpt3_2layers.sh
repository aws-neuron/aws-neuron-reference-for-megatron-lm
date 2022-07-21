#! /bin/bash
set -o pipefail

DATA_PATH=~/examples_datasets/gpt2/my-gpt2_text_document
CHECKPOINT_PATH=~/examples

export MASTER_ADDR='localhost'
export MASTER_PORT=6000
export NEURON_NUM_DEVICES=2

TRAIN_ITERS=1000
if [[ "$NEURON_EXTRACT_GRAPHS_ONLY" == "1" ]]; then
    TRAIN_ITERS=68
fi

python3 pretrain_gpt_mp.py  \
       --tensor-model-parallel-size 2 \
       --num-layers 2 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --micro-batch-size 1 \
       --global-batch-size 1 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters $TRAIN_ITERS \
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
       --tensorboard-dir ./tb_gpt2_2layer_fp32 \
    	|& tee run_log_gpt2_2layer_fp32

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
