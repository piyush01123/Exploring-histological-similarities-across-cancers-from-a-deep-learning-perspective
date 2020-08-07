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



##  Feature Extraction
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch
import argparse
import numpy as np
import h5py
from PIL import Image, ImageFile
import glob
import os
import time

root_dir = '/ssd_scratch/cvit/piyush/KIRC/train'
h5py_file_path = 'test_file.h5'
image_size = 224
batch_size = 64


class h5py_Dataset:
    def __init__(self, file_paths, transform, h5fh):
        self.file_paths = (file_paths)
        self.transform = transform
        h5fh.create_dataset('file_paths', data=np.array(file_paths, dtype='S'))
        slide_ids = [fp.split('/')[-2] for fp in file_paths]
        labels = [0 if fp.split('/')[-3]=='cancer' else 1 for fp in file_paths]
        h5fh.create_dataset('slide_ids', data=np.array(slide_ids, dtype='S'))
        h5fh.create_dataset('labels', data=np.array(labels, dtype=int))
        h5fh.create_dataset('length', data=len(file_paths))
        self.h5fh = h5fh
    def __getitem__(self, idx):
        fp = self.h5fh["file_paths"][idx].decode()
        slide_id = self.h5fh["slide_ids"][idx].decode()
        label = self.h5fh["labels"][idx]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # print(fp, type(fp))
        return self.transform(Image.open(fp)), fp, slide_id, label
    def __len__(self):
        return len(self.file_paths)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
    ])

model = models.resnet18(pretrained=True)
model.fc = Identity()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model).to(device)

if os.path.isfile(h5py_file_path):
    os.remove(h5py_file_path)

h5fh = h5py.File(h5py_file_path, 'w')
## change this if your storage format is different
file_paths = sorted(glob.glob("{}/*/*/*.png".format(root_dir)))
dataset = h5py_Dataset(file_paths=file_paths, transform=transform, h5fh=h5fh)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

for i,(x,y,z,w) in enumerate(dataloader,start=0):
    print(i, x.shape)
    if i==10:
        break


for i,(x,y,z,w) in enumerate(dataloader,start=1000):
    print(i, x.shape)
    if i==1010:
        break


for i,(x,y,z,w) in enumerate(dataloader,start=10000):
    print(i, x.shape)
    if i==10010:
        break
