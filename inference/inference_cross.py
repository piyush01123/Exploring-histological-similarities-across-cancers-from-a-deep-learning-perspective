

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
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image, ImageFile
import json
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_hyperpara(hparams):
    dropouts = []
    hidden_layer_units = []
    for i in range(hparams['n_layers']):
        dropouts.append(hparams['dropout_l{}'.format(i)])
        hidden_layer_units.append(hparams['n_units_l{}'.format(i)])
    optimizer = hparams['optimizer']
    lr = hparams['lr']
    return dropouts,hidden_layer_units,optimizer,lr


def define_model(dropouts,hidden_layer_units,num_classes=2):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    layers = []
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    for i in range(len(dropouts)):
        out_features = hidden_layer_units[i]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = dropouts[i]
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, num_classes))
    model.fc = nn.Sequential(*layers)
    return model


class ModImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(datasets.ImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path


def test(model, test_dataloader, device, writer, export_dir, save_prefix):
    model.eval()
    correct = 0
    y_pred = []
    y_true = []
    score_all =[]
    print("Y=0-->CANCER, Y=1-->NORMAL", flush=True)
    df = pd.DataFrame(columns=['paths', 'slide_ids', 'targets', 'preds', 'probs'])
    with torch.no_grad():
        for batch_id, (data, targets, paths) in enumerate(test_dataloader):
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            score_all.append(output)
            probs = F.softmax(output, dim=1)
            preds = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

            pred_class_probs = probs[torch.arange(len(preds))[:,None],preds].squeeze(1)
            df_batch = pd.DataFrame({'paths': list(paths),
                                     'slide_ids': [p.split('/')[-2] for p in paths],
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

    # Sklearn roc api expects both Y_True and Scores/Probabilities in a matrix of shape MxC where C=num_classes
    # From sklearn docs: "can be either probability estimates or non-thresholded decision value"
    # Y_True have to be in a one-hot encoding format
    # Here we can have 3 roc curves and 3 roc-auc scores, micro, using cancer as positive and using normal as positive
    # Usually the curve and score with cancer as positive is more relevant
    score_all = torch.cat(score_all).cpu().numpy()
    target_all = np.eye(len(classes))[y_true]
    fpr,tpr,roc_auc_dict = {},{},{}
    for i,cls in enumerate(classes):
        fpr[cls], tpr[cls], _ = metrics.roc_curve(target_all[:, i], score_all[:, i])
        roc_auc_dict[cls] = metrics.auc(fpr[cls], tpr[cls])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(target_all.ravel(), score_all.ravel())
    roc_auc_dict["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    for cls in classes+["micro"]:
        plt.figure()
        plt.plot(fpr[cls], tpr[cls], color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc_dict[cls])
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC Curve {} {}".format(save_prefix, cls))
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(export_dir, "{}_{}_ROC_curve.jpg".format(save_prefix,cls)))

    conf_mat = metrics.confusion_matrix(y_true, y_pred).tolist()
    kappa_score = metrics.cohen_kappa_score(y_true, y_pred)

    df.to_csv(os.path.join(export_dir, "{}_record.csv".format(save_prefix)), index=False)
    fh = open(os.path.join(export_dir, "{}_results.txt".format(save_prefix)), 'w')
    fh.write("Classification report\n")
    fh.write(json.dumps(report)+'\n\n')
    fh.write("ROC-AUC scores:\n{}\n\n".format(roc_auc_dict))
    fh.write("Confusion Matrix:\n{}\n\n".format(conf_mat))
    fh.write("Kappa Score:\n{}\n\n".format(kappa_score))
    fh.close()


def main():
    parser = argparse.ArgumentParser(description='Process args for Classifer')
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--log_dir", type=str, default="/ssd_scratch/cvit/ashishmenon/unknown/logs_test/")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--hparam_json", type=str, required=True)
    parser.add_argument("--model_organ", type=str, required=True)
    parser.add_argument("--export_dir", type=str, default='/ssd_scratch/cvit/ashishmenon/unknown/results_test/')
    parser.add_argument("--save_prefix", type=str) # For the organ whose data is being used
    args = parser.parse_args()
    print(args, flush=True)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if not os.path.exists(args.export_dir):
        os.makedirs(args.export_dir)

    writer = SummaryWriter(log_dir=args.log_dir)

    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ])

    test_dataset = ModImageFolder(root=args.test_dir, transform=data_transform)
    nw = 4 if torch.cuda.is_available() else 0
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    hparams = json.load(open(args.hparam_json, 'r'))[args.model_organ]
    dropouts,hidden_layer_units,optimizer_name,lr = get_hyperpara(hparams)
    model = define_model(dropouts,hidden_layer_units,num_classes=len(test_dataloader.dataset.classes))
    print(model, flush=True)

    ckpt = torch.load(args.model_checkpoint)
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)
    print("[MSG] Model loaded from {}".format(args.model_checkpoint), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE {}".format(device), flush=True)
    model = nn.DataParallel(model).to(device)

    test(model, test_dataloader, device, writer, args.export_dir, args.save_prefix)


if __name__=="__main__":
    main()
#usage
# python inference_cross.py \
    #   --model_checkpoint /ssd_scratch/cvit/${username}/${subtype}/model_ckpt/${subtype}_best_model.pth \
    #   --test_dir /ssd_scratch/cvit/${username}/${subtype}/test_data_for_expt \
    #   --export_dir /ssd_scratch/cvit/${username}/${subtype}/results_filtered_patches_inference/ \
    #   --save_prefix ${subtype} \
    #   --log_dir /ssd_scratch/cvit/${username}/${subtype}/logs_test/ | tee ./${subtype}/${subtype}_inference_filtered_patches_log.txt \
