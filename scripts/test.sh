#!/bin/bash
#
# --------------------------
#SBATCH --job-name="TEST"	             # job name
#SBATCH --partition=peregrine-gpu 		 # partition to which job should be submitted
#SBATCH --qos=gpu_debug			  		 # qos type
#SBATCH --nodes=1                 		 # node count
#SBATCH --cpus-per-gpu=6         		 # cpu-cores per gpu (>1 if multi-threaded tasks)
#SBATCH --mem=32G              		     # total memory per node
#SBATCH --gres=gpu:a100-sxm4-80gb:2      # Request 2 GPU (A100 40GB)
#SBATCH --time=00:10:00          	     # total run time limit (HH:MM:SS)

python -c "import torch; import time; print(f'{torch.cuda.device_count()=}, {torch.cuda.is_available()=}'); time.sleep(60);"