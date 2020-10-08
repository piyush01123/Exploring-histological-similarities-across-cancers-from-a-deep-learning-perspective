

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


def get_hyperpara(organ):
    organ_parameters = {}
    # organ_parameters['LUAD'] = {'n_units_l0': 11, 'n_units_l1': 127, 'dropout_l1': 0.21644319500667114, 'lr': 0.000443123860947793, 'n_layers': 3, 'dropout_l0': 0.2710437523297514, 'n_units_l2': 6, 'dropout_l2': 0.4907440672692972, 'optimizer': 'Adam'}
    organ_parameters['LUAD'] = {'dropout_l1': 0.40199979514662076, 'lr': 0.022804111060047597, 'n_layers': 2, 'dropout_l0': 0.2494019459238684, 'n_units_l0': 56, 'optimizer': 'SGD', 'n_units_l1': 72}

    organ_parameters['KICH'] = {'optimizer': 'Adam', 'n_layers': 2, 'lr': 0.00034174604486655424, 'dropout_l1': 0.2590922048730916, 'dropout_l0': 0.4508962491601658, 'n_units_l0': 85, 'n_units_l1': 128}
    organ_parameters['KIRC'] = {'dropout_l0': 0.23949940999396052, 'optimizer': 'Adam', 'lr': 7.340640278726522e-05, 'dropout_l1': 0.23075402550504015, 'n_layers': 2, 'n_units_l0': 90, 'n_units_l1': 60}
    organ_parameters['COAD'] = {'n_layers': 2, 'n_units_l0': 51, 'dropout_l0': 0.3633901781496375, 'n_units_l1': 96, 'dropout_l1': 0.24938970860378834, 'optimizer': 'RMSprop', 'lr': 9.325098201927569e-05}
    organ_parameters['KIRP'] = {'optimizer': 'Adam', 'n_units_l0': 101, 'lr': 0.0001821856779144607, 'n_units_l1': 127, 'dropout_l0': 0.20045233657011846, 'n_layers': 2, 'dropout_l1': 0.24987265944230833}
    organ_parameters['READ'] = {'n_units_l1': 4, 'n_layers': 3, 'n_units_l2': 101, 'optimizer': 'Adam', 'lr': 0.00010345081363438437, 'dropout_l1': 0.32566024092209167, 'n_units_l0': 114, 'dropout_l0': 0.3736031169357898, 'dropout_l2': 0.4320328109269479}
    organ_parameters['LIHC'] = {'lr': 0.000305882784838735, 'n_units_l1': 46, 'n_layers': 2, 'dropout_l1': 0.33086051605996936, 'dropout_l0': 0.3992380294683205, 'n_units_l0': 30, 'optimizer': 'Adam'}
    organ_parameters['LUSC'] = {'dropout_l0': 0.36601040402058993, 'lr': 0.00012063189382769252, 'dropout_l2': 0.23198155500094464, 'dropout_l1': 0.46238255107808696, 'n_units_l1': 5, 'n_units_l0': 128, 'optimizer': 'Adam', 'n_units_l2': 79, 'n_layers': 3}

    dropouts = []
    hidden_layer_units = []
    for i in range(organ_parameters[organ]['n_layers']):
        dropouts.append(organ_parameters[organ]['dropout_l{}'.format(i)])
        hidden_layer_units.append(organ_parameters[organ]['n_units_l{}'.format(i)])
    optimizer = organ_parameters[organ]['optimizer']
    lr = organ_parameters[organ]['lr']
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
        roc_auc_dict[cls] = metrics.auc(fpr[i], tpr[i])
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
        plt.title(f"ROC Curve {save_prefix} {cls}")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(export_dir, f"{save_prefix}_{cls}_ROC_curve.jpg"))

    conf_mat = metrics.confusion_matrix(y_true, y_pred).tolist()
    kappa_score = metrics.cohen_kappa_score(y_true, y_pred)

    df.to_csv(os.path.join(export_dir, f"{save_prefix}_record.csv"), index=False)
    fh = open(os.path.join(export_dir, f"{save_prefix}_results.txt"), 'w')
    fh.write(json.dumps(report))
    fh.write(f"ROC-AUC scores:\n{roc_auc_dict}\n\n")
    fh.write(f"Confusion Matrix:\n{conf_mat}\n\n")
    fh.write(f"Kappa Score:\n{kappa_score}\n\n")
    fh.close()


def main():
    parser = argparse.ArgumentParser(description='Process args for Classifer')
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--log_dir", type=str, default="/ssd_scratch/cvit/ashishmenon/unknown/logs_test/")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--export_dir", type=str, default='/ssd_scratch/cvit/ashishmenon/unknown/results_test/')
    parser.add_argument("--save_prefix", type=str)
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
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=nw, shuffle=False)

    # model = models.resnet18()
    # model.fc = nn.Sequential(
    #             nn.Dropout(p=0.2), # p is prob. of a neuron not being dropped out
    #             nn.Linear(model.fc.in_features, len(test_dataset.classes))
    #             )

    dropouts,hidden_layer_units,optimizer_name,lr = get_hyperpara(args.save_prefix)
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
# python inference.py \
    #   --model_checkpoint /ssd_scratch/cvit/${username}/${subtype}/model_ckpt/${subtype}_best_model.pth \
    #   --test_dir /ssd_scratch/cvit/${username}/${subtype}/test_data_for_expt \
    #   --export_dir /ssd_scratch/cvit/${username}/${subtype}/results_filtered_patches_inference/ \
    #   --save_prefix ${subtype} \
    #   --log_dir /ssd_scratch/cvit/${username}/${subtype}/logs_test/ | tee ./${subtype}/${subtype}_inference_filtered_patches_log.txt \
