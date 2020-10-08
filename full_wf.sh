#!/bin/bash
#SBATCH -A ashishmenon
#SBATCH -n 20
#SBATCH --nodelist=gnode57
#SBATCH --wait-all-nodes=0
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END


envname="T1_task"

source activate ${envname}
username="ashishmenon" 
export PATH=$PATH:/bin
module load cuda/10.0
module load cudnn/7.3-cuda-10.0



for subtype in "BRCA" ;
  do
    echo "Subtype: $subtype"

    echo "Subtype: $subtype"

    echo "Subtype: $subtype"

    echo "Subtype: $subtype"


    mkdir ./${subtype}/
    # rm -rf /ssd_scratch/cvit/${username}/${subtype}/
    mkdir -p /ssd_scratch/cvit/${username}/${subtype}/
    # # BRCA    COAD    KICH    KIRC    KIRP    LIHC    Lists   LUAD    LUSC    PRAD    READ    STAD

    # rsync -zaPq ecdp2020@10.4.16.73:TCGA/${subtype}/ /ssd_scratch/cvit/${username}/${subtype}/SLIDES/
    #rsync -zaPq ecdp2020@10.4.16.73:TCGA_PATCHES/${subtype}/ /ssd_scratch/cvit/${username}/${subtype}/
    # rsync -zaPq ashishmenon@gnode54:/ssd_scratch/cvit/${username}/${subtype}/ /ssd_scratch/cvit/${username}/${subtype}/

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


    # python get_expt_data.py \
    #   --data_dir /ssd_scratch/cvit/${username}/${subtype}/train/ \

    # python get_expt_data.py \
    #   --data_dir /ssd_scratch/cvit/${username}/${subtype}/val/ \

    # python get_expt_data.py \
    #   --data_dir /ssd_scratch/cvit/${username}/${subtype}/test/


    # python list_expt.py --expt_train_dir /ssd_scratch/cvit/${username}/${subtype}/train_data_for_expt/ \
    #       --expt_val_dir /ssd_scratch/cvit/${username}/${subtype}/val_data_for_expt \
    #       --expt_test_dir /ssd_scratch/cvit/${username}/${subtype}/test_data_for_expt/ \
    #       --subtype ${subtype}


    # python move_expt.py \
    #         --train_dir /ssd_scratch/cvit/${username}/${subtype}/train \
    #         --val_dir /ssd_scratch/cvit/${username}/${subtype}/val \
    #         --test_dir /ssd_scratch/cvit/${username}/${subtype}/test \
    #         --expt_train_dir /ssd_scratch/cvit/${username}/${subtype}/train_data_for_expt \
    #         --expt_val_dir /ssd_scratch/cvit/${username}/${subtype}/val_data_for_expt \
    #         --expt_test_dir /ssd_scratch/cvit/${username}/${subtype}/test_data_for_expt \
    #         --expt_json ./${subtype}/${subtype}_expt.json \


    python hyper_para_tuning.py \
      --train_dir /ssd_scratch/cvit/${username}/${subtype}/train_data_for_expt/ \
      --val_dir /ssd_scratch/cvit/${username}/${subtype}/val_data_for_expt/ \
      --save_prefix ${subtype}_hyper_search \
      --log_dir /ssd_scratch/cvit/${username}/${subtype}/logs_hyper_search/ \
      --model_save_path /ssd_scratch/cvit/${username}/${subtype}/model_ckpt_hyper_search/ \
      --num_epochs 20 \
      --num_trials 50 | tee ./${subtype}_hyper_para_tuning/${subtype}_train_log.txt


    # python patch_cnn.py \
    #   --train_dir /ssd_scratch/cvit/${username}/${subtype}/train_data_for_expt/ \
    #   --val_dir /ssd_scratch/cvit/${username}/${subtype}/val_data_for_expt/ \
    #   --save_prefix ${subtype} \
    #   --log_dir /ssd_scratch/cvit/${username}/${subtype}/logs_train/ \
    #   --model_save_path /ssd_scratch/cvit/${username}/${subtype}/model_ckpt/ \
    #   --num_epochs 40  | tee ./${subtype}/${subtype}_train_log_full_filtered_patches.txt


    # python inference.py \
    #   --model_checkpoint /ssd_scratch/cvit/${username}/${subtype}/model_ckpt/${subtype}_best_model.pth \
    #   --test_dir /ssd_scratch/cvit/${username}/${subtype}/test_data_for_expt \
    #   --record_csv /ssd_scratch/cvit/${username}/${subtype}/results_filtered_patches_inference/ \
    #   --save_prefix ${subtype} \
    #   --log_dir /ssd_scratch/cvit/${username}/${subtype}/logs_test/ | tee ./${subtype}/${subtype}_inference_filtered_patches_log.txt \
      

    # python extract_features.py \
    #   --root_dir /ssd_scratch/cvit/${username}/${subtype}/train/ \
    #   --h5py_file_path /ssd_scratch/cvit/${username}/${subtype}/features/${subtype}_train.h5 | tee ./${subtype}/feat_${subtype}_train.txt

    # python extract_features.py \
    #   --root_dir /ssd_scratch/cvit/${username}/${subtype}/val/ \
    #   --h5py_file_path /ssd_scratch/cvit/${username}/${subtype}/features/${subtype}_val.h5 | tee ./${subtype}/feat_${subtype}_val.txt

    # python extract_features.py \
    #   --root_dir /ssd_scratch/cvit/${username}/${subtype}/test/ \
    #   --h5py_file_path /ssd_scratch/cvit/${username}/${subtype}/features/${subtype}_test.h5 | tee ./${subtype}/feat_${subtype}_test.txt


    # python extract_features.py \
    #   --root_dir /ssd_scratch/cvit/${username}/${subtype}/train/ \
    #   --representation_type finetuned \
    #   --model_checkpoint /ssd_scratch/cvit/${username}/${subtype}/model_ckpt/${subtype}_model_epoch_24.pth\
    #   --h5py_file_path /ssd_scratch/cvit/${username}/${subtype}/features/${subtype}_finetuned_train.h5 | tee ./${subtype}/finetuned_feat_${subtype}_train.txt \
      
    # python extract_features.py \
    #   --root_dir /ssd_scratch/cvit/${username}/${subtype}/val/ \
    #   --representation_type finetuned \
    #   --model_checkpoint /ssd_scratch/cvit/${username}/${subtype}/model_ckpt/${subtype}_model_epoch_24.pth\
    #   --h5py_file_path /ssd_scratch/cvit/${username}/${subtype}/features/${subtype}_finetuned_val.h5 | tee ./${subtype}/finetuned_feat_${subtype}_val.txt\
      
    # python extract_features.py \
    #   --root_dir /ssd_scratch/cvit/${username}/${subtype}/test/ \
    #   --representation_type finetuned \
    #   --model_checkpoint /ssd_scratch/cvit/${username}/${subtype}/model_ckpt/${subtype}_model_epoch_24.pth\
    #   --h5py_file_path /ssd_scratch/cvit/${username}/${subtype}/features/${subtype}_finetuned_test.h5 | tee ./${subtype}/finetuned_feat_${subtype}_test.txt \
      

    rsync -zaP /ssd_scratch/cvit/${username}/${subtype}/model_ckpt/${subtype}_best_model.pth ashishmenon@ada:/share3/ashishmenon/Task_T1_results/${subtype}/best_model/
    rsync -zaP /ssd_scratch/cvit/${username}/${subtype}/results_filtered_patches_inference/ ashishmenon@ada:/share3/ashishmenon/Task_T1_results/${subtype}/record_csv/
    rsync -zaP /ssd_scratch/cvit/${username}/${subtype}/logs_train/ ashishmenon@ada:/share3/ashishmenon/Task_T1_results/${subtype}/logs_train/

  # rsync -av --exclude 'test' --exclude 'train' --exclude 'val' --exclude 'PATCHES' --exclude 'SLIDES' --exclude 'PATCHES' \ /ssd_scratch/cvit/${username}/${subtype}/ ashishmenon@ada:/share3/ashishmenon/Task_T1_results/${subtype}/                                                                                                 

      
  done

