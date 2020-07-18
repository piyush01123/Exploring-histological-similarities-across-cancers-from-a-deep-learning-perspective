#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-type=END
#SBATCH --time 1-00:00:00
#SBATCH -w gnode26

module load cuda/9.0
module load cudnn/7-cuda-9.0
source /home/delta_one/v3env/bin/activate

for x in PATCHES PATCHES_KIRP PATCHES_KICH
do
  for y in train valid test
  do
    for z in cancer normal
    do
      echo "Extracting features for ${x} ${y} ${z}"
      python feature_extract.py --img_dir /ssd_scratch/cvit/medicalImaging/${x}/${y}/${z} --npy_file_path /ssd_scratch/cvit/piyush/${x}_${y}_${z}.npy
      echo "RSyncing features for ${x} ${y} ${z}"
      rsync -aPz /ssd_scratch/cvit/piyush/${x}_${y}_${z}.npy delta_one@ada:/share1/delta_one/
      echo "RSync Successful for ${x} ${y} ${z}"
    done
  done
done
