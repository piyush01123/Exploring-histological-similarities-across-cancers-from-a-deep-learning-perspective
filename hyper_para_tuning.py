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
import optuna


MODEL_DICT = {"resnet18" : models.resnet18(pretrained=True),
              "resnet34" : models.resnet34(pretrained=True)
             }

parser = argparse.ArgumentParser(description='Process args for Classifer')
parser.add_argument("--train_dir", type=str, required=True)
parser.add_argument("--val_dir", type=str, required=False)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=15)
parser.add_argument("--imagenet_model", type=str, default="resnet18")
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--log_dir", type=str, default="/ssd_scratch/cvit/ashishmenon/unknown/logs/")
parser.add_argument("--save_prefix", type=str, required=True)
parser.add_argument("--model_save_path", type=str, default='/ssd_scratch/cvit/ashishmenon/unknown')
parser.add_argument("--model_checkpoint", type=str, required=False)
parser.add_argument("--num_trials", type=int, required=False,default=100)


args = parser.parse_args()
print(args, flush=True)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

if not os.path.exists(args.model_save_path):
    os.makedirs(args.model_save_path)



writer = SummaryWriter(log_dir=args.log_dir)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10

def define_model(trial,num_classes=2):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features
    layers.append(nn.Linear(in_features, num_classes))
    model.fc = nn.Sequential(*layers)
    return model

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
        writer.add_scalar('Loss/Train', loss.item(), num_batches*epoch+batch_id)
        writer.add_scalar('Accuracy/Train', correct/batch_size*100, num_batches*epoch+batch_id)
    epoch_acc = running_correct/len(train_dataloader.dataset)*100
    epoch_loss = running_loss/len(train_dataloader.dataset)
    print('Epoch:{}'.format(epoch))
    print('Train epoch loss:{}'.format(epoch_loss))
    print('Train epoch acc:{}'.format(epoch_acc))
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
            # print("Validation Done: [{}/{}]".format((batch_id+1)*len(data), \
            #       len(val_dataloader.dataset)), flush=True)

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
        val_dataloader, criterion, batch_size, learning_rate, num_epochs, save_prefix, model_save_path, device, writer,optimizer_name,lr,trial_num):
    model = handle_trainable_params(model, trainable_modules)

    trainable_params = []
    for module in trainable_modules:
        trainable_params.extend(list(module.parameters()))

    optimizer = getattr(optim, optimizer_name)(trainable_params, lr=lr)
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, factor = 0.2)

    print("DEVICE {}".format(device), flush=True)
    model = nn.DataParallel(model).to(device)
    best_val_acc = 0
    for epoch in range(start_epoch, end_epoch):
        train_acc = train(model, train_dataloader, optimizer, criterion, device, epoch, exp_lr_scheduler, writer)
        torch.save(model.state_dict(), "{}/{}_HPTrial_{}_Ep_{}_{}.pth".format(model_save_path,save_prefix,trial_num,epoch))
        val_acc = test(model, val_dataloader, criterion, device, epoch, writer, save_prefix)
        writer.add_scalars('Epoch wise Accuracy', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
        if val_acc>best_val_acc:
            best_val_acc = val_acc
            # torch.save(model.state_dict(), "{}/{}_best_model.pth".format(model_save_path,save_prefix,))
        # slide_wise_analysis(root=val_dir, model=model, epoch=epoch, \
        #                     transform=data_transforms["val"], device=device, \
        #                     batch_size=batch_size, num_char_slide=60, save_prefix=save_prefix)

    return best_val_acc

def objective(trial):
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
    trainable_module_used=[]
    train_dataset = datasets.ImageFolder(root=args.train_dir, transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(root=args.val_dir, transform=data_transforms["val"])

    sampler, class_weights = get_stratified_sampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)

    model = define_model(trial,num_classes=len(train_dataloader.dataset.classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.from_numpy(class_weights/sum(class_weights)).to(torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    print(optimizer_name,lr)

    trainable_modules = [model.fc, model.layer4, model.layer3, model.layer2, model.layer1, model.bn1, model.conv1]
    val_acc = training_loop(0, args.num_epochs, trainable_modules, model, train_dataloader, \
            val_dataloader, criterion, args.batch_size, args.learning_rate, args.num_epochs, args.save_prefix,args.model_save_path, device, writer,optimizer_name,lr,trial.number)

    return val_acc

if __name__=="__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
