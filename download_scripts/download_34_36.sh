#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 8
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END


source /home/$USER/v3env/bin/activate

MANIFEST_DIR=/home/$USER/project/T1_task/partition_wise_manifests

for i in {34..36}
do
  echo Downloading Partition $i;
  mkdir -p /scratch/piyush/RAW_Partition_$i;
  mkdir -p /scratch/piyush/SVS_Partition_$i;
  gdc-client download -n 10 -m $MANIFEST_DIR/TCGA_partition_$i --dir /scratch/piyush/RAW_Partition_$i/;
  find /scratch/piyush/RAW_Partition_$i/ -name "*.svs" | xargs -I {} bash -c "mv {} /scratch/piyush/SVS_Partition_$i/";
done
