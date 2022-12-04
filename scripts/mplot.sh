
username=piyush
mkdir -p plots

BRCA_best="BRCA  COAD  LIHC  READ"
COAD_best="BRCA  COAD  LIHC  READ"
KICH_best="COAD  KICH  KIRP  LIHC  READ"
KIRC_best="BRCA  KICH  KIRC  KIRP  LIHC"
KIRP_best="COAD  KICH  KIRP  LIHC  READ"
LIHC_best="BRCA  COAD  LIHC  READ"
LUAD_best="BRCA  COAD  LIHC  LUAD  LUSC  READ"
LUSC_best="BRCA  LIHC  LUAD  LUSC  READ"
PRAD_best="BRCA  PRAD"
READ_best="COAD  LIHC  READ"
STAD_best="BRCA  COAD  LIHC  READ  STAD"

for model in BRCA      COAD    KICH    KIRC    KIRP    LIHC    LUAD    LUSC    PRAD    READ    STAD
do
  for method in PCA KernelPCA LDA TSNE MDS Isomap SpectralEmbedding LLE LTSA HessianLLE ModifiedLLE
  do
    best=${model}_best
    python plot_cross_emb.py \
          --h5py_files_root /ssd_scratch/cvit/${username}/Features/${model}_Model/ \
          --test_sets ${!best} \
          --model_organ ${model} \
          --points_to_use 50 \
          --method ${method} \
          --outfile ./plots/Plot_${model}_${method}_Best.jpg

  python plot_cross_emb.py \
          --h5py_files_root /ssd_scratch/cvit/${username}/Features/${model}_Model/ \
          --test_sets BRCA	COAD	KICH	KIRC	KIRP	LIHC	LUAD	LUSC	PRAD	READ	STAD \
          --model_organ ${model} \
          --points_to_use 50 \
          --method ${method} \
          --outfile ./plots/Plot_${model}_${method}_All.jpg
  done
done

