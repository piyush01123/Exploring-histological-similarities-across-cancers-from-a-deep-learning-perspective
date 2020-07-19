

source ~/v3env/bin/activate
python patch_cnn.py \
  --h5py_file_path_train /ssd_scratch/cvit/piyush/KIRC/KIRC_train.h5 \
  --h5py_file_path_val /ssd_scratch/cvit/piyush/KIRC/KIRC_val.h5 \
  --val_dir /ssd_scratch/cvit/piyush/KIRC/val/ \
  --log_dir logs_dense/ \
  --save_prefix KIRC_Dense \
  --num_epochs 12  | tee kirc_train_log_dense.txt
