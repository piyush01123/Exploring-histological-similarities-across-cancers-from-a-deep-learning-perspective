

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
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class h5py_Dataset:
    def __init__(self, h5py_file_path):
        self.h5fh = h5py.File(h5py_file_path, 'r')
        self.classes = ["cancer", "normal"]

    def __getitem__(self, idx):
        embedding = self.h5fh["embeddings"][idx]
        label = self.h5fh["labels"][idx]
        return torch.from_numpy(embedding.astype(np.float32)), label

    def __len__(self):
        return self.h5fh["length"].value


class DenseModel(nn.Module):
    def __init__(self):
        super(DenseModel, self).__init__()
        self.fc1 = nn.Linear(512,128)
        self.fc2 = nn.Linear(128,2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def test(model, test_dataloader, device, writer, record_csv):
    model.eval()
    correct = 0
    y_pred = []
    y_true = []
    print("Y=0-->CANCER, Y=1-->NORMAL", flush=True)
    df = pd.DataFrame(columns=['paths', 'slide_ids', 'targets', 'preds', 'probs'])
    with torch.no_grad():
        for batch_id, (embeddings, paths, slide_ids, targets) in enumerate(test_dataloader):
            data, targets = embeddings.to(device), targets.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

            pred_class_probs = probs[torch.arange(len(preds))[:,None],preds].squeeze(1)
            df_batch = pd.DataFrame({'paths': list(paths),
                                     'slide_ids': list(slide_ids),
                                     'targets': targets.tolist(),
                                     'preds': preds.squeeze(1).tolist(),
                                     'probs': pred_class_probs.tolist(),
                                    })
            df = df.append(df_batch)
            y_pred.append(output.argmax(dim=1))
            y_true.append(targets)
            correct += preds.eq(targets.view_as(preds)).sum().item()
            print('[{}/{}] Done'.format((batch_id+1)*len(data),len(test_dataloader.dataset)), flush=True)
    test_acc = 100.*correct/len(test_dataloader.dataset)
    print("Test set: Accuracy: {}/{} ({:.2f}%)".format(correct, len(test_dataloader.dataset), \
          test_acc), flush=True)

    y_true, y_pred = torch.cat(y_true).cpu().numpy(), torch.cat(y_pred).cpu().numpy()
    classes = test_dataloader.dataset.classes
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, digits=4)
    print(report, flush=True)
    df.to_csv(record_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description='Process args for Classifer')
    parser.add_argument("--test_h5py_file_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--record_csv", type=str, required=True)
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.log_dir)

    test_dataset = h5py_Dataset(h5py_Dataset=args.test_h5py_file_path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = DenseModel()
    ckpt = torch.load(args.model_checkpoint)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    print("[MSG] Model loaded from {}".format(args.model_checkpoint), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE {}".format(device), flush=True)
    model = nn.DataParallel(model).to(device)

    test(model, test_dataloader, device, writer, args.record_csv)


if __name__=="__main__":
    main()
