
source /home/delta_one/v3env/bin/activate

python extract_features.py \
  --root_dir /ssd_scratch/cvit/piyush/KIRC/train \
	--h5py_file_path /ssd_scratch/cvit/piyush/KIRC_train.h5 | tee feat_kirc_train.txt

python extract_features.py \
  --root_dir /ssd_scratch/cvit/piyush/KIRC/val \
  --h5py_file_path /ssd_scratch/cvit/piyush/KIRC_val.h5 | tee feat_kirc_val.txt


python extract_features.py \
  --root_dir /ssd_scratch/cvit/piyush/KIRC/test \
	--h5py_file_path /ssd_scratch/cvit/piyush/KIRC_test.h5 | tee feat_kirc_test.txt
