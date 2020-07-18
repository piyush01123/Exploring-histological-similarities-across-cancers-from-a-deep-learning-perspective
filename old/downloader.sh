#!/bin/bash
#SBATCH -A delta_one
#SBATCH --reservation=non-deadline-queue
#SBATCH -n 10
#SBATCH --nodelist gnode43


username="delta_one"

subtype_name=$1

mkdir -p /ssd_scratch/cvit/${username}/${subtype_name}/
mkdir -p /ssd_scratch/cvit/${username}/${subtype_name}/slides/
mkdir -p /ssd_scratch/cvit/${username}/${subtype_name}/patches/


cd /ssd_scratch/cvit/${username}/${subtype_name}/

## create a file named KICH.txt at ~/
cp /home/${username}/${subtype_name}.txt .

source /home/${username}/v2env/bin/activate
gdc-client download -m ${subtype_name}.txt -n 20

rm ${subtype_name}.txt

find /ssd_scratch/cvit/${username}/${subtype_name}/ -name '*.svs' | xargs mv -t /ssd_scratch/cvit/${username}/${subtype_name}/slides/

deactivate

python3 /home/${username}/patch_extractor.py /ssd_scratch/cvit/${username}/${subtype_name}/
setfacl -m u:ashishmenon:rwx /ssd_scratch/cvit/${username}/
