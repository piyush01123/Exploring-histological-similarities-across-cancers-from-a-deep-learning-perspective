#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 28
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-type=END
#SBATCH --time 1-00:00:00
#SBATCH -w gnode22


module load cuda/9.0
module load cudnn/7-cuda-9.0
source /home/delta_one/v3env/bin/activate

for y in train valid test
do
  for z in KIRC KIRP KICH
	do
    echo "Extracting features for ${y} ${z}"
    python feature_extract.py --img_dir /ssd_scratch/cvit/medicalImaging/subtype_classification/${y}/${z}/cancer \
															--npy_file_path ${y}_${z}_epoch_13.npy \
                              --batch_size 24 \
															--model_checkpoint /home/delta_one/project/histopathology/exports/subtype_FC_Layer_4_1/SUBTYPE_model_epoch_13.pth
  done
done
