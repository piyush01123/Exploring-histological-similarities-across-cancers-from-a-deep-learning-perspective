#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 16
##SBATCH --wait-all-nodes=0
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END

module load cuda/10.0
module load cudnn/7.3-cuda-10.0

subtype="STAD"
source ~/v3env/bin/activate


mkdir -p /ssd_scratch/cvit/piyush/${subtype}/SLIDES

mkdir -p ./${subtype}_hyper_para_tuning/


# rsync -zaPq ecdp2020@10.4.16.73:TCGA/${subtype}/ /ssd_scratch/cvit/piyush/${subtype}/SLIDES/
rsync -zaPq ecdp2020@10.4.16.73:TCGA/${subtype}/ /ssd_scratch/cvit/piyush/${subtype}/SLIDES/

cd ..

python extract_patches.py \
  --root_dir /ssd_scratch/cvit/piyush/${subtype}/SLIDES/ \
  --dest_dir /ssd_scratch/cvit/piyush/${subtype}/PATCHES/ | tee ./${subtype}/log_patch.txt

python divide.py \
  --subtype ${subtype} \
  --root_dir /ssd_scratch/cvit/piyush/${subtype}/PATCHES/ \
  --train_dir /ssd_scratch/cvit/piyush/${subtype}/train/  \
  --val_dir /ssd_scratch/cvit/piyush/${subtype}/val/  \
  --test_dir /ssd_scratch/cvit/piyush/${subtype}/test/



python get_expt_data.py \
  --data_dir /ssd_scratch/cvit/piyush/${subtype}/train/ \

python get_expt_data.py \
  --data_dir /ssd_scratch/cvit/piyush/${subtype}/val/ \

python get_expt_data.py \
  --data_dir /ssd_scratch/cvit/piyush/${subtype}/test/

# rsync -zaPq ashishmenon@gnode43:/ssd_scratch/cvit/piyush/${subtype}/test_data_for_expt /ssd_scratch/cvit/piyush/${subtype}/
# rsync -zaPq ashishmenon@gnode43:/ssd_scratch/cvit/piyush/${subtype}/val_data_for_expt /ssd_scratch/cvit/piyush/${subtype}/
# rsync -zaPq ashishmenon@gnode43:/ssd_scratch/cvit/piyush/${subtype}/train_data_for_expt /ssd_scratch/cvit/piyush/${subtype}/



python hyper_para_tuning.py \
  --train_dir /ssd_scratch/cvit/piyush/${subtype}/train_data_for_expt/ \
  --val_dir /ssd_scratch/cvit/piyush/${subtype}/val_data_for_expt/ \
  --save_prefix ${subtype}_hyper_search \
  --log_dir /ssd_scratch/cvit/piyush/${subtype}/logs_hyper_search/ \
  --model_save_path /ssd_scratch/cvit/piyush/${subtype}/model_ckpt_hyper_search/ \
  --num_epochs 15 \
  --num_trials 50 | tee ./${subtype}_hyper_para_tuning/${subtype}_train_log.txt
