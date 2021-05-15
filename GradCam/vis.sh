#!/bin/bash

#SBATCH -A ashishmenon
#SBATCH -n 36
##SBATCH --nodelist=gnode13
##SBATCH --wait-all-nodes=0
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END

username=ashishmenon
module load cuda/10.0
module load cudnn/7.3-cuda-10.0

mkdir -p /ssd_scratch/cvit/ashishmenon/


## ---- Best cases ------ ##

# BRCA_best="BRCA  COAD  LIHC  READ"
# COAD_best="BRCA  COAD  LIHC  READ"
# LIHC_best="BRCA  COAD  LIHC  READ"
# READ_best="COAD  LIHC  READ"


# KICH_best="COAD  KICH  KIRP  LIHC  READ"
# KIRP_best="COAD  KICH  KIRP  LIHC  READ"
# KIRC_best="BRCA  KICH  KIRC  KIRP  LIHC"


# PRAD_best="BRCA READ LIHC"


# LUAD_best="BRCA  COAD  LIHC  LUSC  READ"
# LUSC_best="BRCA  LIHC  LUAD  READ"
# STAD_best="BRCA  COAD  LIHC  READ"

## ---- worst cases ------ ##

# BRCA_worst="KICH KIRC"
# COAD_best="KICH KIRC "
# LIHC_worst="KICH KIRC"
# READ_worst="KIRC KICH"
# PRAD_worst="KICH KIRC"
# STAD_worst="KICH KIRC"
# LUAD_worst="KICH KIRC"
# LUSC_worst="KICH KIRC"

# KICH_worst="KIRC STAD"
# KIRP_worst="PRAD STAD"
# KIRC_worst="PRAD STAD"





for model in KICH KIRC
do
  rsync -zaPq ashishmenon@gnode20:/ssd_scratch/cvit/ashishmenon/${model} /ssd_scratch/cvit/ashishmenon/ 
  rsync -zaPq ashishmenon@gnode20:/ssd_scratch/cvit/ashishmenon/visualization_output_4/${model}_${model} /ssd_scratch/cvit/ashishmenon/visualization_output_4/ 
done

for model in PRAD STAD 
do
  rsync -zaPq ashishmenon@gnode13:/ssd_scratch/cvit/ashishmenon/${model} /ssd_scratch/cvit/ashishmenon/ 
  rsync -zaPq ashishmenon@gnode13:/ssd_scratch/cvit/ashishmenon/visualization_output_4/${model}_${model} /ssd_scratch/cvit/ashishmenon/visualization_output_4/ 
done 

rsync -zaPq ecdp2020@10.4.16.73:T1/CheckPoints/ /ssd_scratch/cvit/ashishmenon/ckpts/



#--------------------------------------------Setting up of data ----------------------------------------------------------------------#

# for model in PRAD STAD  
# do
#   rsync -zaPq ecdp2020@10.4.16.73:TCGA_PATCHES/${model}/test /ssd_scratch/cvit/${username}/${model}/
#   rsync -zaPq ecdp2020@10.4.16.73:T1/CheckPoints/${model}_best_model.pth /ssd_scratch/cvit/ashishmenon/ckpts/
#   rsync -zaPq ecdp2020@10.4.16.73:ExptDataJson/${model}_expt.json /ssd_scratch/cvit/${username}/expt_data_json/
 
#   python3 /home/ashishmenon/T1_task_aux/move_expt.py  --test True \
#                                                       --test_dir /ssd_scratch/cvit/${username}/${model}/test      \
#                                                       --expt_test_dir /ssd_scratch/cvit/${username}/${model}/test_data_for_expt/ \
#                                                       --expt_json /ssd_scratch/cvit/${username}/expt_data_json/${model}_expt.json

#   python save_best_cancer_samples.py --test_dir /ssd_scratch/cvit/ashishmenon/${model}/test_data_for_expt/ \
#                                      --model_checkpoint /ssd_scratch/cvit/ashishmenon/ckpts/${model}_best_model.pth \
#                                      --hparam_json ../best_hparams.json \
#                                      --save_dir ../Best_cancer_samples/ \
#                                      --model_chosen ${model}
#   rm -rf /ssd_scratch/cvit/${username}/${model}/test
# done

#------------------------------------------------------------------------------------------------------------------------------------#




#Predictions
for inference in KIRC KICH
do
  for model in BRCA COAD LIHC READ PRAD STAD LUAD LUSC KICH
  do
    if [ ${inference} = ${model} ];then
     continue
    fi
    python grad_cam_vis.py \
      --test_dir /ssd_scratch/cvit/ashishmenon/${inference}/test_data_for_expt/ \
      --model_checkpoint /ssd_scratch/cvit/ashishmenon/ckpts/${model}_best_model.pth \
      --hparam_json best_hparams.json \
      --save_dir /ssd_scratch/cvit/ashishmenon/visualization_output_4/${model}_${inference}/ \
      --model_chosen ${model} \
      --inferred_on ${inference} 
  done  
done



for inference in STAD PRAD
do
  for model in KICH KIRP KIRC
  do
    if [ ${inference} = ${model} ];then
     continue
    fi
    python grad_cam_vis.py \
      --test_dir /ssd_scratch/cvit/ashishmenon/${inference}/test_data_for_expt/ \
      --model_checkpoint /ssd_scratch/cvit/ashishmenon/ckpts/${model}_best_model.pth \
      --hparam_json best_hparams.json \
      --save_dir /ssd_scratch/cvit/ashishmenon/visualization_output_4/${model}_${inference}/ \
      --model_chosen ${model} \
      --inferred_on ${inference} 
  done  
done


#------------------------------------------------------------------------------------------------------------------------------------#


#Ground truth generation

# for inference in BRCA READ LIHC COAD LUAD LUSC STAD
# do
#   for model in BRCA READ LIHC COAD LUAD LUSC STAD 
#   do
#   if [ ${inference} != ${model} ];then
#     continue
#   fi
#     python grad_cam_vis.py \
#       --test_dir /ssd_scratch/cvit/ashishmenon/${inference}/test_data_for_expt/ \
#       --model_checkpoint /ssd_scratch/cvit/ashishmenon/ckpts/${model}_best_model.pth \
#       --hparam_json best_hparams.json \
#       --save_dir /ssd_scratch/cvit/ashishmenon/visualization_output_4/${model}_${inference}/ \
#       --model_chosen ${model} \
#       --inferred_on ${inference} 
#   done  
# done


#------------------------------------------------------------------------------------------------------------------------------------#


#Finding IOU and Jaccard
for inference in KIRC KICH
do
  for model in BRCA COAD LIHC READ PRAD STAD LUAD LUSC KICH
  do 
    if [ ${inference} = ${model} ];then
      continue
    fi
    python save_jaccard_iou.py \
       --save_dir /ssd_scratch/cvit/ashishmenon/IOU_inferences/ \
       --vis_output_dir /ssd_scratch/cvit/ashishmenon/visualization_output_4 \
       --model ${model} \
       --infer_on ${inference}
  done 
done

for inference in STAD PRAD
do
  for model in KICH KIRP KIRC
  do 
    if [ ${inference} = ${model} ];then
      continue
    fi
    python save_jaccard_iou.py \
       --save_dir /ssd_scratch/cvit/ashishmenon/IOU_inferences/ \
       --vis_output_dir /ssd_scratch/cvit/ashishmenon/visualization_output_4 \
       --model ${model} \
       --infer_on ${inference}
  done
  
done



#------------------------------------------------------------------------------------------------------------------------------------#