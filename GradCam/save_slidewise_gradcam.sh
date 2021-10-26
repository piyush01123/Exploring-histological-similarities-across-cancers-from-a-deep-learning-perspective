#!/bin/bash

#SBATCH -A research
#SBATCH -n 08
#SBATCH --nodelist=gnode41
#SBATCH --wait-all-nodes=0
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END

username=ashishmenon
module load cuda/10.0
module load cudnn/7.3-cuda-10.0
source activate T1_task
mkdir -p /ssd_scratch/cvit/ashishmenon/

# rsync -aPq ashishmenon@gnode50:/ssd_scratch/cvit/ashishmenon/visualization_output_4 /ssd_scratch/cvit/ashishmenon/
# rsync -aPq ashishmenon@gnode50:/ssd_scratch/cvit/ashishmenon/IOU_inferences /ssd_scratch/cvit/ashishmenon/

python save_slide_wise_grad_cam_try_2.py

