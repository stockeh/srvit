#!/bin/bash
#
# --------------------------
#SBATCH --job-name="GREMLIN-ViT Eval"	 # job name
#SBATCH --partition=peregrine-gpu 		 # partition to which job should be submitted
#SBATCH --qos=gpu_short          		 # qos type
#SBATCH --nodes=1                 		 # node count
#SBATCH --cpus-per-gpu=6         		 # cpu-cores per gpu (>1 if multi-threaded tasks)
#SBATCH --mem=64G              		     # total memory per node
#SBATCH --gres=gpu:a100-sxm4-80gb:1      # Request GPU (A100 80GB)
#SBATCH --time=00:30:00          	     # total run time limit (HH:MM:SS)
#SBATCH --output=../logs/log.out         # output log file
#SBATCH --error=../logs/log.err      	 # error file
#SBATCH --mail-type=all        	         # send email 
#SBATCH --mail-user=stock@colostate.edu

python ../src/main.py -m vit --cuda -b 16 --workers 8 --test --save