#!/bin/bash
#SBATCH -A ashishmenon
#SBATCH -n 10
#SBATCH --nodelist=gnode46
#SBATCH --wait-all-nodes=0
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END


username=ashishmenon
module load cuda/10.0
module load cudnn/7.3-cuda-10.0


# for subtype in  COAD  KIRC  KIRP  
# do
#   python3 ../T1_task_aux/move_expt.py  --test True \
#                         --test_dir /ssd_scratch/cvit/${username}/${subtype}/test      \
#                         --expt_test_dir /ssd_scratch/cvit/${username}/${subtype}/test_data_for_expt/ \
#                         --expt_json /ssd_scratch/cvit/${username}/expt_data_json/${subtype}_expt.json
# done


# mkdir -p /ssd_scratch/cvit/${username}/CheckPoints
for model in LUAD LIHC LUSC READ BRCA COAD KICH KIRP KIRC PRAD STAD   
do
  # rsync -zaPq ecdp2020@10.4.16.73:TCGA_PATCHES/${model}/test /ssd_scratch/cvit/${username}/${model}/
  python3 move_expt.py  --test True \
                        --test_dir /ssd_scratch/cvit/${username}/${model}/test      \
                        --expt_test_dir /ssd_scratch/cvit/${username}/${model}/test_data_for_expt/ \
                        --expt_json /ssd_scratch/cvit/${username}/expt_data_json/${model}_expt.json

  for subtype in ${model} LIHC  KICH  KIRC  KIRP  
  do
    python grad_cam_vis.py \
      --test_dir /ssd_scratch/cvit/ashishmenon/${subtype}/test_data_for_expt/ \
      --model_checkpoint /ssd_scratch/cvit/ashishmenon/ckpts/${model}_best_model.pth \
      --hparam_json best_hparams.json \
      --save_dir /ssd_scratch/cvit/ashishmenon/visualization_output/${model}_${subtype}/ \
      --model_chosen ${model} \
      --inferred_on ${subtype}
    done
done


