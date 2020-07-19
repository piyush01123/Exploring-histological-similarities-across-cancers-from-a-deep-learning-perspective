

source /home/delta_one/v3env/bin/activate

python inference_dense.py \
  --test_h5py_file_path /ssd_scratch/cvit/piyush/KIRC_test.h5 \
  --model_checkpoint checkpoints/KIRC_Dense_model_epoch_11.pth \
  --record_csv records_dense.csv \
  --log_dir inf_dense_logs/ | tee kirc_inference_dense_log.txt
