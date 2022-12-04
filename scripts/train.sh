#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 32
##SBATCH --wait-all-nodes=0
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END

username=piyush
subtype=STAD

module load cuda/10.0
module load cudnn/7.3-cuda-10.0
source ~/v3env/bin/activate
mkdir -p /ssd_scratch/cvit/${username}/ExptDataJson/
mkdir -p /ssd_scratch/cvit/${username}/${subtype}/
rsync -aPzq gnode19:/ssd_scratch/cvit/${username}/${subtype}/  /ssd_scratch/cvit/${username}/${subtype}/
rsync -aPz ecdp2020@10.4.16.73:ExptDataJson/${subtype}_expt.json /ssd_scratch/cvit/${username}/ExptDataJson/

# python3 expt_data/merge_expt.py \
#         --train_dir /ssd_scratch/cvit/${username}/${subtype}/train \
#         --val_dir /ssd_scratch/cvit/${username}/${subtype}/val \
#         --test_dir /ssd_scratch/cvit/${username}/${subtype}/test \
#         --expt_train_dir /ssd_scratch/cvit/${username}/${subtype}/train_data_for_expt \
#         --expt_val_dir /ssd_scratch/cvit/${username}/${subtype}/val_data_for_expt \
#         --expt_test_dir /ssd_scratch/cvit/${username}/${subtype}/test_data_for_expt \
#         --expt_json /ssd_scratch/cvit/${username}/ExptDataJson/${subtype}_expt.json
#

python3 expt_data/move_expt.py \
        --train_dir /ssd_scratch/cvit/${username}/${subtype}/train \
        --val_dir /ssd_scratch/cvit/${username}/${subtype}/val \
        --test_dir /ssd_scratch/cvit/${username}/${subtype}/test \
        --expt_train_dir /ssd_scratch/cvit/${username}/${subtype}/train_data_for_expt \
        --expt_val_dir /ssd_scratch/cvit/${username}/${subtype}/val_data_for_expt \
        --expt_test_dir /ssd_scratch/cvit/${username}/${subtype}/test_data_for_expt \
        --expt_json /ssd_scratch/cvit/${username}/ExptDataJson/${subtype}_expt.json

python patch_cnn.py \
      --hparam_json best_hparams.json \
      --train_dir /ssd_scratch/cvit/${username}/${subtype}/train_data_for_expt/ \
      --val_dir /ssd_scratch/cvit/${username}/${subtype}/val_data_for_expt/ \
      --save_prefix ${subtype} \
      --log_dir /ssd_scratch/cvit/${username}/logs_train/ \
      --model_save_path /ssd_scratch/cvit/${username}/model_ckpt/ \
      --num_epochs 40  | tee ${subtype}_train_log_best_hp.txt

mkdir -p ~/${subtype}
cp -r /ssd_scratch/cvit/${username}/${subtype}/model_ckpt/ ~/{subtype}/
