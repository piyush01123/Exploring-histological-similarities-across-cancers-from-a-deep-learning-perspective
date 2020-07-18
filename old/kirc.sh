#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-type=END
#SBATCH --time 1-00:00:00
#SBATCH -w gnode22

module load cuda/9.0
module load cudnn/7-cuda-9.0
source /home/delta_one/v3env/bin/activate

python /home/delta_one/project/histopathology/classifier.py \
                     --img_dir /ssd_scratch/cvit/medicalImaging/PATCHES/train/ \
                     --val_dir /ssd_scratch/cvit/medicalImaging/PATCHES/valid/ \
                     --num_epochs 10 \
                     --log_dir kirc_logs/  \
                     --batch_size 64 \
                     --model_checkpoint kirc_checkpoints/KIRC_model_epoch_5.pth \
                     --optimzer_checkpoint kirc_checkpoints/KIRC_optimizer_epoch_5.pth \
                     --exp_lr_scheduler kirc_checkpoints/KIRC_exp_lr_scheduler_epoch_5.pth \
                     --save_prefix KIRC > trg_log_kirc.txt
sleep 5
