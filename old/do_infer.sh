#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 2
#SBATCH --gres gpu:0
#SBATCH --nodes 1
#SBATCH --mem-per-cpu 1024
#SBATCH --mail-type END
#SBATCH --time 1-00:00:00
#SBATCH -w gnode28

## module load cuda/9.0
## module load cudnn/7-cuda-9.0
source /home/delta_one/v3env/bin/activate

python /home/delta_one/project/rcc_classification/inference.py \
                    --model_checkpoint exports/kirc_epoch_6_to_8_FC/KIRC_model_epoch_8.pth \
                    --test_dir /ssd_scratch/cvit/PATCHES/ \
                    --log_dir kirc_logs > kirc_inference_upsampled_log.txt
sleep 5
