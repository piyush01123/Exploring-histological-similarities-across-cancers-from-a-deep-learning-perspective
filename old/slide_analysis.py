

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


def roc_curve(y_true, y_pred):
    fpr = []
    tpr = []
    acc = []
    thresholds = np.arange(0.0, 1.01, .01)

    P = sum(y_true)
    N = len(y_true) - P

    for thresh in thresholds:
        FP=0
        TP=0
        for y_t, y_p in zip(y_true, y_pred):
            if y_p >= thresh:
                if y_t == 1:
                    TP = TP + 1
                else:
                    FP = FP + 1
        fpr.append(FP/float(N))
        tpr.append(TP/float(P))
        acc.append((TP+N-FP)/len(y_true))
    # plt.plot(fpr, tpr)
    return tpr, fpr, acc, thresholds


def integrate(x_s, y_s):
    # integartion by trapezoidal rule
    x_diffs = [abs(x_s[i]-x_s[i-1]) for i in range(1,len(x_s))]
    y_sum = [y_s[i]+y_s[i-1] for i in range(1,len(y_s))]
    return .5*sum([a*b for a, b in zip(y_sum, x_diffs)])


class SlideDataset(Dataset):
    def __init__(self, file_paths, transform):
        super(SlideDataset, self).__init__()
        self.file_paths = file_paths
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(Image.open(self.file_paths[idx]))

    def __len__(self):
        return len(self.file_paths)


def slide_wise_analysis(root, model, epoch, classes, transform, device, batch_size, num_char_slide, save_prefix):
    model.eval()
    isImage = lambda f: f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")

    slide_wise_output = dict()
    results_gt = []
    results_pred = []
    for class_id, class_ in enumerate(classes):
        slide_wise_output[class_] = dict()
        files = os.listdir(os.path.join(root, class_))
        files = [f for f in files if isImage(f)]
        slide_ids = set([f[:num_char_slide] for f in files])

        for slide_id in slide_ids:
            y_pred = []
            slide_files = [os.path.join(root, class_, f) for f in files if f.startswith(slide_id)]
            slide_files = sorted(slide_files)
            slide_dataset = SlideDataset(slide_files, transform)
            slide_dataloader = DataLoader(slide_dataset, batch_size=batch_size, shuffle=False)
            for batch in slide_dataloader:
                batch = batch.to(device)
                out = model(batch)
                y_pred.append(out.argmax(dim=1))

            y_pred = torch.cat(y_pred)
            fraction_positive = int(sum(y_pred).cpu().numpy())/len(y_pred)
            slide_wise_output[class_][slide_id] = fraction_positive

            results_gt.append(class_id)
            results_pred.append(fraction_positive)

    print("slide_wise_output", slide_wise_output, flush=True)
    print("results_gt", results_gt, flush=True)
    print("results_pred", results_pred, flush=True)

    tpr, fpr, accuracies, thresholds = roc_curve(results_gt, results_pred)
    roc_auc = integrate(x_s=fpr, y_s=tpr)

    acc_max = max(accuracies)
    threshold_opt = thresholds[accuracies.index(acc_max)]

    print("Slide Wise ROC-AUC = {}".format(roc_auc), flush=True)
    roc_auc = metrics.roc_auc_score(results_gt, results_pred)
    print("Slide Wise ROC-AUC by sklearn = {}".format(roc_auc), flush=True)
    print("Max Accuracy = {} at fraction = {}".format(acc_max, threshold_opt), flush=True)

    conf_mat = metrics.confusion_matrix(results_gt, results_pred>threshold_opt)
    print("Confusion Matrix\n", conf_mat, flush=True)
    kappa_score = metrics.cohen_kappa_score(results_gt, results_pred>threshold_opt)
    print("Cohen Kappa Score = {}".format(kappa_score), flush=True)


    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.set_title('Receiver Operating Characteristic')
    ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    plt.savefig("{}_slide_roc_curve_epoch_{}.jpg".format(save_prefix, epoch))

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.set_title('Accuracy vs Threshold')
    ax.plot(thresholds, accuracies, 'b', label = 'max acc = %0.2f at T =%0.2f' % \
            (max(accuracies), thresholds[accuracies.index(max(accuracies))]))
    ax.legend(loc = 'lower right')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Threshold Fraction of normal patches for WSI to be considered Normal')
    plt.savefig("{}_accuracy_vs_threshold_epoch_{}.jpg".format(save_prefix, epoch))
