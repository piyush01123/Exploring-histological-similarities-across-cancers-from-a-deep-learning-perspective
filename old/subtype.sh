#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 32
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-type=END
#SBATCH --time 1-00:00:00
#SBATCH -w gnode22

module load cuda/9.0
module load cudnn/7-cuda-9.0
source /home/delta_one/v3env/bin/activate

python /home/delta_one/project/histopathology/classifier.py \
                     --train_dir /ssd_scratch/cvit/medicalImaging/subtype_classification/train/ \
                     --val_dir /ssd_scratch/cvit/medicalImaging/subtype_classification/valid/ \
                     --batch_size 64 \
                     --num_epochs 30 \
                     --log_dir subtype_logs/  \
                     --model_checkpoint exports/subtype_FC_with_dropout/SUBTYPE_model_epoch_3.pth \
                     --save_prefix SUBTYPE > trg_log_subtype.txt
sleep 5
