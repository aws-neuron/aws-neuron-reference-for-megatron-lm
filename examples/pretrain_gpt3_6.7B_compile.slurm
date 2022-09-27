#!/bin/bash
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
 
srun neuron_parallel_compile ./examples/pretrain_gpt3_6.7B_32layers_bf16_bs1024_slurm.sh
