

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
import os


MODEL_DICT = {"resnet18" : models.resnet18(pretrained=True),
              "resnet34" : models.resnet34(pretrained=True)
             }

def train(model, train_dataloader, optimizer, criterion, device, epoch, scheduler, writer):
    num_batches = len(train_dataloader)
    model.train()
    running_correct = 0
    running_loss = 0
    for batch_id, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        optimizer.zero_grad()
        loss = criterion(input=output, target=target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        running_correct += correct
        batch_size = len(data)
        running_loss += loss.item()*batch_size
        print("[Train] Epoch: {} [{}/{}]    Loss: {:.6f}   Batch Acc: {:.2f}".format(
              epoch, (batch_id+1)*batch_size, len(train_dataloader.dataset),
              loss.item(), correct/batch_size*100), flush=True)
        writer.add_scalar('Loss/Train', loss.item(), num_batches*epoch+batch_id)
        writer.add_scalar('Accuracy/Train', correct/batch_size*100, num_batches*epoch+batch_id)
    epoch_acc = running_correct/len(train_dataloader.dataset)*100
    epoch_loss = running_loss/len(train_dataloader.dataset)
    scheduler.step(epoch_loss)
    return epoch_acc


def test(model, val_dataloader, criterion, device, epoch, writer, save_prefix):
    model.eval()
    val_loss = 0
    correct = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(val_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            y_pred.append(output.argmax(dim=1))
            y_true.append(target)
            correct += pred.eq(target.view_as(pred)).sum().item()
            print("Validation Done: [{}/{}]".format((batch_id+1)*len(data), \
                  len(val_dataloader.dataset)), flush=True)

    val_loss /= len(val_dataloader.dataset)

    val_acc = 100.*correct/len(val_dataloader.dataset)
    print("[Test] Epoch: {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
          epoch, val_loss, correct, len(val_dataloader.dataset), val_acc), flush=True
         )
    writer.add_scalar('Loss/Eval', val_loss, epoch)
    writer.add_scalar('Accuracy/Eval', val_acc, epoch)
    y_true, y_pred = torch.cat(y_true).cpu().numpy(), torch.cat(y_pred).cpu().numpy()

    classes = val_dataloader.dataset.classes
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, digits=4)
    print(report, flush=True)
    return val_acc


def get_stratified_sampler(img_dataset):
    imgs, targets, num_classes = img_dataset.imgs, np.array(img_dataset.targets), len(img_dataset.classes)
    class_counts = np.empty(num_classes,)
    for i in range(num_classes):
        class_counts[i] = sum(targets==i)
    class_weights = sum(class_counts)/class_counts # N/count not count/N
    sample_weights = torch.empty((len(imgs),), dtype=torch.double)
    for i in range(num_classes):
        sample_weights[targets==i] = class_weights[i]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler, class_weights


def handle_trainable_params(model, trainable_modules):
    # layers need to model.module objects
    for param in model.parameters():
        param.requires_grad = False
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True
    return model


def training_loop(start_epoch, end_epoch, trainable_modules, model, train_dataloader, \
        val_dataloader, criterion, batch_size, learning_rate, num_epochs, save_prefix, device, writer):
    model = handle_trainable_params(model, trainable_modules)

    trainable_params = []
    for module in trainable_modules:
        trainable_params.extend(list(module.parameters()))

    optimizer = optim.Adam(params=trainable_params, lr=learning_rate, weight_decay=0.05)
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, factor = 0.2)

    print("DEVICE {}".format(device), flush=True)
    model = nn.DataParallel(model).to(device)

    for epoch in range(start_epoch, end_epoch):
        train_acc = train(model, train_dataloader, optimizer, criterion, device, epoch, exp_lr_scheduler, writer)
        torch.save(model.state_dict(), "{}_model_epoch_{}.pth".format(save_prefix, epoch))
        val_acc = test(model, val_dataloader, criterion, device, epoch, writer, save_prefix)
        writer.add_scalars('Epoch wise Accuracy', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
        # slide_wise_analysis(root=val_dir, model=model, epoch=epoch, \
        #                     transform=data_transforms["val"], device=device, \
        #                     batch_size=batch_size, num_char_slide=60, save_prefix=save_prefix)


def main():
    parser = argparse.ArgumentParser(description='Process args for Classifer')
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--imagenet_model", type=str, default="resnet18")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--save_prefix", type=str, required=True)
    parser.add_argument("--model_checkpoint", type=str, required=False)

    args = parser.parse_args()
    print(args, flush=True)

    writer = SummaryWriter(log_dir=args.log_dir)

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ]),
    }

    train_dataset = datasets.ImageFolder(root=args.train_dir, transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(root=args.val_dir, transform=data_transforms["val"])

    sampler, class_weights = get_stratified_sampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    model = MODEL_DICT[args.imagenet_model]
    model.fc = nn.Sequential(
                nn.Dropout(p=0.2), # p is prob. of a neuron not being dropped out
                nn.Linear(model.fc.in_features, len(train_dataset.classes))
                )
    print(model, flush=True)

    if args.model_checkpoint is not None:
        # IMP: This is because checkpoint dictionary has "module." in each key
        ckpt = torch.load(args.model_checkpoint)
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        print("[MSG] Model loaded from {}".format(args.model_checkpoint), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.from_numpy(class_weights/sum(class_weights)).to(torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_modules = [model.fc, model.layer4]
    print("Training Layers fc, layer4", flush=True)
    training_loop(0, args.num_epochs//3, trainable_modules, model, train_dataloader, \
            val_dataloader, criterion, args.batch_size, args.learning_rate, args.num_epochs, args.save_prefix, device, writer)

    print("Training Layers fc, layer4, layer3, layer2", flush=True)
    trainable_modules = [model.fc, model.layer4, model.layer3, model.layer2]
    training_loop(args.num_epochs//3, 2*args.num_epochs//3, trainable_modules, model, train_dataloader, \
            val_dataloader, criterion, args.batch_size, args.learning_rate, args.num_epochs, args.save_prefix, device, writer)

    print("Training Layers fc, layer4, layer3, layer2, layer1, bn1, conv1", flush=True)
    trainable_modules = [model.fc, model.layer4, model.layer3, model.layer2, model.layer1, model.bn1, model.conv1]
    training_loop(2*args.num_epochs//3, args.num_epochs, trainable_modules, model, train_dataloader, \
            val_dataloader, criterion, args.batch_size, args.learning_rate, args.num_epochs, args.save_prefix, device, writer)



if __name__=="__main__":
    main()
