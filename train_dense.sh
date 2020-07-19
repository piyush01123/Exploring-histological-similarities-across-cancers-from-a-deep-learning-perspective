

source ~/v3env/bin/activate
python dense_nn_features.py \
  --h5py_file_path_train /ssd_scratch/cvit/piyush/KIRC_train.h5 \
  --h5py_file_path_val /ssd_scratch/cvit/piyush/KIRC_val.h5 \
  --log_dir logs_dense/ \
  --save_prefix KIRC_Dense \
  --num_epochs 12  | tee kirc_train_log_dense.txt
