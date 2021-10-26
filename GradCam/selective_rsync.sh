#!/bin/bash
mkdir -p /ssd_scratch/cvit/ashishmenon/test_patches/$1/
echo "First arg: $1"
echo "Second arg: $2"

rsync --dry-run -aP   ecdp2020@10.4.16.73:TCGA/$1/ /ssd_scratch/cvit/ashishmenon/ | grep $2 | xargs -I {} sh -c  "rsync -aPq ecdp2020@10.4.16.73:TCGA/$1/{} /ssd_scratch/cvit/ashishmenon/$1/"
rsync -aPq   ecdp2020@10.4.16.73:TCGA_PATCHES/$1/test/cancer/$2 /ssd_scratch/cvit/ashishmenon/test_patches/$1/ 