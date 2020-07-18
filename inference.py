

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torch
import torchvision
from PIL import Image
import argparse
import copy
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ModImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(datasets.ImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path


def test(model, test_dataloader, device, writer):
    model.eval()
    correct = 0
    y_pred = []
    y_true = []
    print("Y=0-->CANCER, Y=1-->NORMAL", flush=True)
    with torch.no_grad():
        for batch_id, (data, target, paths) in enumerate(test_dataloader):

            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

            for (i_path, i_prob, i_y, i_pred) in zip(paths, probs, target, preds):
                print("path=\"{}\", prob={}, y={}, pred={}".format(i_path, i_prob.cpu().numpy()[1], i_y, i_pred.cpu().numpy()[1]), flush=True)

            y_pred.append(output.argmax(dim=1))
            y_true.append(target)
            correct += preds.eq(target.view_as(preds)).sum().item()
    test_acc = 100.*correct/len(test_dataloader.dataset)
    print("Test set: Accuracy: {}/{} ({:.2f}%)".format(correct, len(test_dataloader.dataset), \
          test_acc), flush=True)

    y_true, y_pred = torch.cat(y_true).cpu().numpy(), torch.cat(y_pred).cpu().numpy()
    classes = test_dataloader.dataset.classes
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, digits=4)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Process args for Classifer')
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.log_dir)

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ])

    test_dataset = ModImageFolder(root=args.test_dir, transform=data_transform)
    nw = 4 if torch.cuda.is_available() else 0
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=nw, shuffle=False)

    model = models.resnet18()
    model.fc = nn.Sequential(
                nn.Dropout(p=0.2), # p is prob. of a neuron not being dropped out
                nn.Linear(model.fc.in_features, len(test_dataset.classes))
                )

    ckpt = torch.load(args.model_checkpoint)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    print("[MSG] Model loaded from {}".format(args.model_checkpoint), flush=True)

    print("DEVICE {}".format(device), flush=True)
    model = nn.DataParallel(model).to(device)


    test(model, test_dataloader, device, writer)


if __name__=="__main__":
    main()
