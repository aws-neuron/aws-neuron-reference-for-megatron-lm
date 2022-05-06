#!/usr/bin/env bash
set -o pipefail

TRAIN_SCRIPT=$1

workdir=workdir
if [ -e $workdir ]; then
        echo "$NL$INFO: Removing existing workdir $workdir"
        rm -rf $workdir
fi
mkdir -p $workdir

cd $workdir

neuron_parallel_compile sh ../$TRAIN_SCRIPT |& tee parallel_compile.txt

if [ $? -eq 0 ]; then
        success=1
else
        success=0
fi

if [ $success -eq 1 ]; then
        sh ../$TRAIN_SCRIPT |& tee train_log.txt
else 
	echo "Parallel Compile failed. Not running training script."
fi
