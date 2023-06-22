#!/bin/bash
#
#SBATCH --job-name=ai2es_js
#SBATCH -p a100
#SBATCH --account=ai2es_premium
#SBATCH -t 14-00:00                  # time limit: (D-HH:MM) 
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-gpu=16            # --cpus-per-task=8
#SBATCH --mem=128gb                  # --mem-per-cpu=1gb
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --output=../logs/log_%j.out  # output log file
#SBATCH --error=../logs/log_%j.err   # error file
#SBATCH --mail-type=all        	     # send email 
#SBATCH --mail-user=stock@colostate.edu

apptainer exec --cleanenv --nv --bind /home/jstock:/home/jstock --bind /rdma/dgx-a100/jstock:/home/jstock/data /home/jstock/apptainer/DRIVE_PYTORCH/code_server_pytorch.sif /home/jstock/miniconda/envs/dev/bin/python ../src/main.py -e unet01 -m unet --cuda --epochs 300 -b 64 --workers 8 --data-dir /home/jstock/data/conus3/A/ --lr 0.0001 --loss genexp --unet-hiddens 32 32 32
