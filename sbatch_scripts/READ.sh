#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 10
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END


SUBTYPE=READ

mkdir -p /ssd_scratch/cvit/TCGA/"$SUBTYPE"/SLIDES/
mkdir -p /ssd_scratch/cvit/TCGA/"$SUBTYPE"/PATCHES/
mkdir -p /ssd_scratch/cvit/TCGA/"$SUBTYPE"/train/
mkdir -p /ssd_scratch/cvit/TCGA/"$SUBTYPE"/val/
mkdir -p /ssd_scratch/cvit/TCGA/"$SUBTYPE"/test/

rsync -aPz ecdp2020@10.4.16.73:/mnt/base/ecdp2020/TCGA/"$SUBTYPE"/ /ssd_scratch/cvit/TCGA/"$SUBTYPE"/SLIDES/

source /home/$USER/v3env/bin/activate
python extract_patches.py \
  --root_dir /ssd_scratch/cvit/TCGA/"$SUBTYPE"/SLIDES/ \
  --dest_dir /ssd_scratch/cvit/TCGA/"$SUBTYPE"/PATCHES/  | tee log_patch_"$SUBTYPE".txt

python divide.py \
  --subtype "$SUBTYPE" \
  --root_dir /ssd_scratch/cvit/TCGA/"$SUBTYPE"/PATCHES/ \
  --train_dir /ssd_scratch/cvit/TCGA/"$SUBTYPE"/train/ \
  --val_dir /ssd_scratch/cvit/TCGA/"$SUBTYPE"/val/ \
  --test_dir /ssd_scratch/cvit/TCGA/"$SUBTYPE"/test/ \
  --ratio .7 .2 .1

python extract_features.py \
  --root_dir /ssd_scratch/cvit/TCGA/"$SUBTYPE"/train \
	--h5py_file_path /ssd_scratch/cvit/TCGA/"$SUBTYPE"_train.h5 | tee feat_"$SUBTYPE"_train.txt

python extract_features.py \
  --root_dir /ssd_scratch/cvit/TCGA/"$SUBTYPE"/val \
  --h5py_file_path /ssd_scratch/cvit/TCGA/"$SUBTYPE"_val.h5 | tee feat_"$SUBTYPE"_val.txt

python extract_features.py \
  --root_dir /ssd_scratch/cvit/TCGA/"$SUBTYPE"/test \
	--h5py_file_path /ssd_scratch/cvit/TCGA/"$SUBTYPE"_test.h5 | tee feat_"$SUBTYPE"_test.txt

python patch_cnn.py \
  --train_dir /ssd_scratch/cvit/TCGA/"$SUBTYPE"/train/ \
  --val_dir /ssd_scratch/cvit/TCGA/"$SUBTYPE"/val/ \
  --save_prefix "$SUBTYPE" \
  --log_dir "$SUBTYPE"_log_trn  \
  --num_epochs 25  | tee "$SUBTYPE"_train_log.txt

last_epoch=24
mkdir -p checkpoints
cp "$SUBTYPE"_model_epoch_"$last_epoch".pth checkpoints/

python inference.py \
  --model_checkpoint checkpoints/"$SUBTYPE"_model_epoch_"$last_epoch".pth \
  --test_dir /ssd_scratch/cvit/TCGA/"$SUBTYPE"/test \
  --log_dir "$SUBTYPE"_log_inf  \
  --record_csv "$SUBTYPE"_record.csv | tee "$SUBTYPE"_inference_log.txt

python aggregate_slide_wise.py \
  --record_file "$SUBTYPE"_record.csv \
	--dest_dir "$SUBTYPE"_aggregate

echo "$SUBTYPE" FINISHED.
