#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 16
##SBATCH --wait-all-nodes=0
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END

username=piyush
module load cuda/10.0
module load cudnn/7.3-cuda-10.0
source ~/v3env/bin/activate

rm -rf /ssd_scratch/cvit/${username}
mkdir -p /ssd_scratch/cvit/${username}/Dummy
mkdir -p /ssd_scratch/cvit/${username}/ExptDataJson

for subtype in BRCA	COAD	KICH	KIRC	KIRP	LIHC	LUAD	LUSC	PRAD	READ	STAD
do
  mkdir -p /ssd_scratch/cvit/${username}/${subtype}/test/
  rsync -aPz ecdp2020@10.4.16.73:TCGA_PATCHES/${subtype}/test/ /ssd_scratch/cvit/${username}/${subtype}/test/
  rsync -aPz ecdp2020@10.4.16.73:ExptDataJson/${subtype}_expt.json /ssd_scratch/cvit/${username}/ExptDataJson/

  python expt_data/move_expt.py \
          --train_dir /ssd_scratch/cvit/${username}/Dummy \
          --val_dir /ssd_scratch/cvit/${username}/Dummy \
          --test_dir /ssd_scratch/cvit/${username}/${subtype}/test \
          --expt_train_dir /ssd_scratch/cvit/${username}/Dummy \
          --expt_val_dir /ssd_scratch/cvit/${username}/Dummy \
          --expt_test_dir /ssd_scratch/cvit/${username}/${subtype}/test_data_for_expt \
          --expt_json /ssd_scratch/cvit/${username}/ExptDataJson/${subtype}_expt.json
done


mkdir -p /ssd_scratch/cvit/${username}/CheckPoints
for model in BRCA	COAD	KICH	KIRC	KIRP	LIHC	LUAD	LUSC	PRAD	READ	STAD
do
  ckpt=${model}_best_model.pth
  mkdir -p ~/CrossOrganInference/${model}_Model
  rsync -aPz ecdp2020@10.4.16.73:CheckPoints/${ckpt} /ssd_scratch/cvit/${username}/CheckPoints/
  for subtype in BRCA	COAD	KICH	KIRC	KIRP	LIHC	LUAD	LUSC	PRAD	READ	STAD
  do
    echo Model=${model}, TestData=${subtype}
    python inference_cross.py \
          --model_checkpoint /ssd_scratch/cvit/${username}/CheckPoints/${ckpt} \
          --hparam_json best_hparams.json \
          --test_dir /ssd_scratch/cvit/${username}/${subtype}/test_data_for_expt \
          --export_dir /ssd_scratch/cvit/${username}/CrossOrganInference/${model}_Model/${subtype} \
          --save_prefix ${subtype} \
          --model_organ ${model} \
          --log_dir /ssd_scratch/cvit/${username}/Logs_Test/${model}_Model/${subtype}/ | tee Crs_Inf_${model}_${subtype}.txt
    done
done

cp -r /ssd_scratch/cvit/${username}/CrossOrganInference ~/
tar cvzf ~/CrossOrgan.tar.gz ~/CrossOrganInference/
