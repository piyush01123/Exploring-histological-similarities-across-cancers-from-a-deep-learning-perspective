username=piyush
source /home/delta_one/v3env/bin/activate
model=BRCA
python tsne.py \
        --h5py_files_root /ssd_scratch/cvit/${username}/Features/${model}_Model/ \
        --test_sets BRCA	COAD	LIHC	READ \
        --model_organ ${model} \
        --points_to_use 50 \
        --outfile ./TSNE_Plot_${model}.jpg
