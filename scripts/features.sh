
username=piyush
source /home/delta_one/v3env/bin/activate

# python extract_features.py \
#   --root_dir /ssd_scratch/cvit/piyush/KIRC/train \
# 	--h5py_file_path /ssd_scratch/cvit/piyush/KIRC_train.h5 | tee feat_kirc_train.txt
#
# python extract_features.py \
#   --root_dir /ssd_scratch/cvit/piyush/KIRC/val \
#   --h5py_file_path /ssd_scratch/cvit/piyush/KIRC_val.h5 | tee feat_kirc_val.txt
#
#
# python extract_features.py \
#   --root_dir /ssd_scratch/cvit/piyush/KIRC/test \
# 	--h5py_file_path /ssd_scratch/cvit/piyush/KIRC_test.h5 | tee feat_kirc_test.txt


for model in BRCA	COAD	KICH	KIRC	KIRP	LIHC	LUAD	LUSC	PRAD	READ	STAD
do
  ckpt=${model}_best_model.pth
  mkdir -p /ssd_scratch/cvit/${username}/Features/${model}_Model
  for testset in BRCA	COAD	KICH	KIRC	KIRP	LIHC	LUAD	LUSC	PRAD	READ	STAD
  do
    echo Model=${model}, TestData=${testset}
    python feature_extractor.py \
          --root_dir /ssd_scratch/cvit/${username}/${testset}/test_data_for_expt/ \
          --model_checkpoint /ssd_scratch/cvit/${username}/CheckPoints/${model}_best_model.pth \
          --hparam_json best_hparams.json \
          --model_organ ${model} \
          --h5py_file_path /ssd_scratch/cvit/piyush/Features/${model}_Model/${testset}.h5
    done
done
