## Dataloader and sampler test

# from torchvision import transforms, datasets
import numpy as np, torch
tdir = '/ssd_scratch/cvit/piyush/KIRC/train'
image_size=224
t = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
    ])
img_dataset = datasets.ImageFolder(root=tdir, transform=t)
imgs, targets, num_classes = img_dataset.imgs, np.array(img_dataset.targets), len(img_dataset.classes)
class_weights = np.empty(num_classes,)
for i in range(num_classes):
    class_weights[i] = sum(targets==i)
class_weights = class_weights/sum(class_weights)
sample_weights = torch.empty((len(imgs),), dtype=torch.double)
for i in range(num_classes):
    sample_weights[targets==i] = class_weights[i]
sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))


# ------------------------------------------------------------------------------ #

## Reconstruction
import numpy as np
import glob, os
import argparse
from PIL import Image
import cv2
import time
from multiprocessing import Pool
import multiprocessing
import itertools
import pandas as pd

root_dir_cancer = '/ssd_scratch/cvit/piyush/KIRC/test/cancer'
dest_dir = '/ssd_scratch/cvit/piyush/KIRC/heatmaps'
thumbnail_dir = '/ssd_scratch/cvit/piyush/KIRC/heatmap_thumbnails'
patch_df = pd.read_csv('KIRC_stats.csv')
record_df = pd.read_csv('record.csv')
cancer_slide_ids = os.listdir()
slide_class = "cancer"
patch_size = 512
patch_df["slide_id"] = patch_df.slide.apply(lambda x: x.split('.')[0])
cancer_slide_ids = os.listdir(root_dir_cancer)
slide_id = cancer_slide_ids[0]
_,_,W,H,_= patch_df[patch_df.slide_id==slide_id].values.squeeze(0)
im_slide = np.full((W//4,H//4,3),240,dtype=np.uint8)
for x in range(0,W//4,patch_size):
    for y in range(0,H//4,patch_size):
        fp = os.path.join(root_dir_cancer, slide_id, "{}_X_{}_Y_{}.png".format(slide_id, x, y))
        if not os.path.isfile(fp):
            continue
        patch = Image.open(fp)
        patch_arr = np.swapaxes(np.array(patch),1,0)
        im_slide[x:x+patch_size,y:y+patch_size,:] = patch_arr

im_slide_full = Image.fromarray(np.swapaxes(im_slide,1,0))
im_slide_full.save("{}.png".format(slide_id))
im_slide_thumbnail = im_slide_full.resize((1024,int(H/W*1024)))
im_slide_thumbnail.save("thumbnail_{}.png".format(slide_id))


# ------------------------------------------------------------------------------ #

## Heatmap
import numpy as np
import glob, os
import argparse
from PIL import Image
import cv2
import time
from multiprocessing import Pool
import multiprocessing
import itertools
import pandas as pd

root_dir_cancer = '/ssd_scratch/cvit/piyush/KIRC/test/cancer'
dest_dir = '/ssd_scratch/cvit/piyush/KIRC/heatmaps'
thumbnail_dir = '/ssd_scratch/cvit/piyush/KIRC/heatmap_thumbnails'
patch_df = pd.read_csv('KIRC_stats.csv')
record_df = pd.read_csv('record.csv')
cancer_slide_ids = os.listdir()
slide_class = "cancer"
patch_size = 512
patch_df["slide_id"] = patch_df.slide.apply(lambda x: x.split('.')[0])
cancer_slide_ids = os.listdir(root_dir_cancer)
slide_id = cancer_slide_ids[0]
_,_,W,H,_= patch_df[patch_df.slide_id==slide_id].values.squeeze(0)
im_slide = np.full((W//4,H//4,3),240,dtype=np.uint8)
alpha = 0.75

record_subset = record_df[record_df.slide_ids==slide_id]
probs = record_subset.probs
record_subset.probs = (probs-probs.min())/(probs.max()-probs.min())

for x in range(0,W//4,patch_size):
    for y in range(0,H//4,patch_size):
        fp = os.path.join(root_dir_cancer, slide_id, "{}_X_{}_Y_{}.png".format(slide_id, x, y))
        if not os.path.isfile(fp):
            continue
        patch = Image.open(fp)
        patch_arr = np.swapaxes(np.array(patch),1,0)
        prob = record_subset[record_subset.paths==fp].probs.values[0]
        r, g, b = int(prob*255), int((1-prob)*255), 0
        color_tile = np.empty((patch_size,patch_size,3), dtype=np.uint8)
        color_tile[:,:,0] = r
        color_tile[:,:,1] = g
        color_tile[:,:,2] = b
        overlap_tile = (alpha*patch_arr + (1-alpha)*color_tile).astype(np.uint8)
        im_slide[x:x+patch_size,y:y+patch_size,:] = overlap_tile


im_slide_full = Image.fromarray(np.swapaxes(im_slide,1,0))
im_slide_full.save("OLP_{}.png".format(slide_id))

im_slide_thumbnail = im_slide_full.resize((1024,int(H/W*1024)))
im_slide_thumbnail.save("OLP_thumbnail_{}.png".format(slide_id))


# YCbCr
for x in range(0,W//4,patch_size):
    for y in range(0,H//4,patch_size):
        fp = os.path.join(root_dir_cancer, slide_id, "{}_X_{}_Y_{}.png".format(slide_id, x, y))
        if not os.path.isfile(fp):
            continue
        break
    if os.path.isfile(fp):
        break

patch = Image.open(fp)
patch_arr = np.swapaxes(np.array(patch.convert('YCbCr')),1,0)
prob = record_subset[record_subset.paths==fp].probs.values[0]
Cb, Cr = int((1-prob)*255), int(prob*255),
Y = patch_arr[:,:,0]
overlap_tile = np.empty((patch_size,patch_size,3), dtype=np.uint8)
overlap_tile[:,:,0] = Y
overlap_tile[:,:,1] = Cb
overlap_tile[:,:,2] = Cr

Image.fromarray(np.swapaxes(overlap_tile,1,0), mode='YCbCr').save('overlap_tile.jpg')
