

import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import argparse
import numpy as np
from PIL import Image, ImageFile
import glob
import os

parser = argparse.ArgumentParser(description='Process args for Feature Extraction')
parser.add_argument("--img_dir", type=str, required=True)
parser.add_argument("--npy_file_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--imagenet_model", type=str, default="resnet18")
parser.add_argument("--model_checkpoint", type=str, required=True)
parser.add_argument("--image_size", type=int, default=224)


MODEL_DICT = {"resnet18" : models.resnet18(pretrained=True)}
NUM_CLASSES = 3

class ImageDataset(Dataset):
    def __init__(self, file_paths, transform):
        super(ImageDataset, self).__init__()
        self.file_paths = file_paths
        self.transform = transform

    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        return self.transform(Image.open(self.file_paths[idx]))

    def __len__(self):
        return len(self.file_paths)


def feature_extraction(model, device, dataloader, batch_size, npy_file_path):
    model.eval()
    output = np.empty((len(dataloader.dataset), 512))
    for i, batch in enumerate(dataloader):
        batch = batch.to(device)
        out = model(batch)
        out = out.reshape((-1, 512))
        output[i*batch_size : (i+1)*batch_size] = out.cpu().numpy()
        if i%100==0:
            print("[Done]: {}/{}".format((i+1)*batch_size, len(dataloader.dataset)), flush=True)
    np.save(npy_file_path, output)


def main():
    args = parser.parse_args()

    img_dir = args.img_dir
    batch_size = args.batch_size
    image_size = args.image_size
    imagenet_model = args.imagenet_model
    npy_file_path = args.npy_file_path
    model_checkpoint = args.model_checkpoint

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ])

    file_paths = sorted(glob.glob("{}/*/*.png".format(img_dir)))
    dataset = ImageDataset(file_paths=file_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = MODEL_DICT[imagenet_model]
    model.fc = nn.Sequential(
                nn.Dropout(p=0.2), # p is prob. of a neuron not being dropped out
                nn.Linear(model.fc.in_features, NUM_CLASSES)
                )
    ckpt = torch.load(model_checkpoint)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    model = nn.Sequential(*list(model.children())[:-1])

    for param in model.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)
    print("Extracting images from {} at {}".format(img_dir, npy_file_path), flush=True)
    feature_extraction(model, device, dataloader, batch_size, npy_file_path)
    print("FIN.", flush=True)


if __name__=="__main__":
    main()
