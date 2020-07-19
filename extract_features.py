

import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import argparse
import numpy as np
import h5py
from PIL import Image, ImageFile
import glob
import os


MODEL_DICT = {"resnet18" : models.resnet18(pretrained=True)}


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
        print("TYPES", type(self.transform(Image.open(fp))), fp, slide_id, label)
        return self.transform(Image.open(fp)), fp, slide_id, label

    def __len__(self):
        return len(self.file_paths)


def extract_features(model, device, dataloader, batch_size, h5fh):
    model.eval()
    output = np.empty((len(dataloader.dataset), 512))
    with torch.no_grad():
        for i, (batch,_,_,_) in enumerate(dataloader):
            batch = batch.to(device)
            out = model(batch)
            out = out.reshape((-1, 512))
            output[i*batch_size : (i+1)*batch_size] = out.cpu().numpy()
            if i%100==0:
                print("[Done]: {}/{}".format((i+1)*batch_size, len(dataloader.dataset)), flush=True)
        h5fh.create_dataset('embeddings', data=output)


def main():
    parser = argparse.ArgumentParser(description='Process args for Feature Extraction')
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--h5py_file_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--imagenet_model", type=str, default="resnet18")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()


    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ])

    model = MODEL_DICT[args.imagenet_model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)

    h5fh = h5py.File(args.h5py_file_path, 'w')
    ## change this if your storage format is different
    file_paths = sorted(glob.glob("{}/*/*/*.png".format(args.root_dir)))
    dataset = h5py_Dataset(file_paths=file_paths, transform=transform, h5fh=h5fh)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("Extracting images from {} at {}".format(args.root_dir, args.h5py_file_path), flush=True)
    extract_features(model, device, dataloader, args.batch_size, h5fh)
    h5fh.close()
    print("FIN.", flush=True)


if __name__=="__main__":
    main()
