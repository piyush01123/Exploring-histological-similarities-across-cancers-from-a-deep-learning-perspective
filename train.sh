

source ~/v3env/bin/activate
python patch_cnn.py \
  --train_dir /ssd_scratch/cvit/piyush/KIRC/train/ \
  --val_dir /ssd_scratch/cvit/piyush/KIRC/val/ \
  --save_prefix KIRC | tee kirc_train_log.txt
