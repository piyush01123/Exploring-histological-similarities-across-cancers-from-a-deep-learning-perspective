#!/bin/bash
#SBATCH -A ashishmenon
#SBATCH -n 16
##SBATCH --nodelist=gnode11
##SBATCH --wait-all-nodes=0
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END



module load cuda/10.0
module load cudnn/7.3-cuda-10.0

username="ashishmenon"
subtype="KIRC"

envname="T1_task"

source activate ${envname}


# conda install -y -c conda-forge optuna
mkdir -p /ssd_scratch/cvit/${username}/${subtype}/

mkdir -p ./${subtype}_hyper_para_tuning/


# rsync -zaPq ecdp2020@10.4.16.73:TCGA/${subtype}/ /ssd_scratch/cvit/${username}/${subtype}/SLIDES/
rsync -zaPq ecdp2020@10.4.16.73:TCGA_PATCHES/${subtype}/ /ssd_scratch/cvit/${username}/${subtype}/

# python extract_patches.py \
#   --root_dir /ssd_scratch/cvit/${username}/${subtype}/SLIDES/ \
#   --dest_dir /ssd_scratch/cvit/${username}/${subtype}/PATCHES/ \
#   --extras_dir /ssd_scratch/cvit/${username}/${subtype}/EXTRAS/  | tee ./${subtype}/log_patch.txt

# python divide.py \
#   --subtype ${subtype} \
#   --root_dir /ssd_scratch/cvit/${username}/${subtype}/PATCHES/ \
#   --train_dir /ssd_scratch/cvit/${username}/${subtype}/train/  \
#   --val_dir /ssd_scratch/cvit/${username}/${subtype}/val/  \
#   --test_dir /ssd_scratch/cvit/${username}/${subtype}/test/



python get_expt_data.py \
  --data_dir /ssd_scratch/cvit/${username}/${subtype}/train/ \

python get_expt_data.py \
  --data_dir /ssd_scratch/cvit/${username}/${subtype}/val/ \

python get_expt_data.py \
  --data_dir /ssd_scratch/cvit/${username}/${subtype}/test/   

# rsync -zaPq ashishmenon@gnode43:/ssd_scratch/cvit/${username}/${subtype}/test_data_for_expt /ssd_scratch/cvit/${username}/${subtype}/ 
# rsync -zaPq ashishmenon@gnode43:/ssd_scratch/cvit/${username}/${subtype}/val_data_for_expt /ssd_scratch/cvit/${username}/${subtype}/ 
# rsync -zaPq ashishmenon@gnode43:/ssd_scratch/cvit/${username}/${subtype}/train_data_for_expt /ssd_scratch/cvit/${username}/${subtype}/ 



# python hyper_para_tuning.py \
#   --train_dir /ssd_scratch/cvit/${username}/${subtype}/train_data_for_expt/ \
#   --val_dir /ssd_scratch/cvit/${username}/${subtype}/val_data_for_expt/ \
#   --save_prefix ${subtype}_hyper_search \
#   --log_dir /ssd_scratch/cvit/${username}/${subtype}/logs_hyper_search/ \
#   --model_save_path /ssd_scratch/cvit/${username}/${subtype}/model_ckpt_hyper_search/ \
#   --num_epochs 15 \
#   --num_trials 50 | tee ./${subtype}_hyper_para_tuning/${subtype}_train_log.txt