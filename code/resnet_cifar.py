"""
resnet_cifar.py
- Implements ResNet-18 (BasicBlock) adapted to CIFAR-10 (no torchvision.models)
- Train / validate loops, checkpointing, and visualization output saved to runs/cls/
Run:
  python code/resnet_cifar.py --epochs 80 --batch-size 128 --lr 0.1
"""
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, utils

# Model: BasicBlock + ResNet18 adapted for CIFAR
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        # 3x3 conv - BN - ReLU - 3x3 conv - BN - add - ReLU
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out, inplace=True)
        return out

class ResNetCIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        # CIFAR-10: use 3x3 conv, no initial 7x7 or maxpool
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # global avg pool
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18_cifar(num_classes=10):
    return ResNetCIFAR(BasicBlock, [2,2,2,2], num_classes=num_classes)

# Utility: train, validate, save artifacts
def get_dataloaders(batch_size):
    # standard CIFAR-10 mean/std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return trainloader, testloader, trainset, testset

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds==y).sum().item()
        total += x.size(0)
    return running_loss/total, correct/total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss=0.0
    correct=0
    total=0
    all_preds=[]
    all_targets=[]
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds==y).sum().item()
            total += x.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    return running_loss/total, correct/total, np.concatenate(all_preds), np.concatenate(all_targets)

# Visualization helpers
def plot_curves(history, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='train')
    plt.plot(epochs, history['val_loss'], label='val')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], label='train')
    plt.plot(epochs, history['val_acc'], label='val')
    plt.title('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'curves_cls.png'))
    plt.close()

def save_confusion_matrix(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i, format(cm[i,j],fmt),
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_path)
    plt.close()

def save_prediction_grids(model, dataset, device, out_dir, n=16):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=True)
    images, labels = next(iter(loader))
    images_n = images.to(device)
    with torch.no_grad():
        logits = model(images_n)
        preds = logits.argmax(dim=1).cpu()
    # undo normalization for display
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    imgs = images.numpy().transpose(0,2,3,1)
    imgs = (imgs*std + mean).clip(0,1)
    fig = plt.figure(figsize=(8,8))
    for i in range(min(n, len(imgs))):
        ax = fig.add_subplot(4,4,i+1)
        ax.imshow(imgs[i])
        ax.axis('off')
        ax.set_title(f"P:{preds[i]} T:{labels[i].item()}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'preds_grid.png'))
    plt.close()

# Grad-CAM (simple) - uses last conv layer
def gradcam_for_image(model, input_tensor, target_class, device, last_conv_name='layer4'):
    # register hook on last conv features and gradient
    features = []
    grads = []
    def forward_hook(module, inp, out):
        features.append(out.detach())
    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0].detach())
    # locate last conv module object
    last_conv = getattr(model, last_conv_name)
    handle_f = last_conv.register_forward_hook(forward_hook)
    handle_b = last_conv.register_full_backward_hook(backward_hook)
    model.eval()
    input_tensor = input_tensor.to(device).unsqueeze(0)
    input_tensor.requires_grad = True
    logits = model(input_tensor)
    score = logits[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=False)
    fmap = features[0][0].cpu().numpy()  # C,H,W
    g = grads[0][0].cpu().numpy()
    weights = g.mean(axis=(1,2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i]
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    # cleanup hooks
    handle_f.remove()
    handle_b.remove()
    return cam  # HxW normalized

def save_gradcam_images(model, dataset, device, out_dir, num=4):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    for i, (img, label) in enumerate(loader):
        if i>=num: break
        img_np = img.numpy().transpose(0,2,3,1)[0]
        img_vis = (img_np*std + mean).clip(0,1)
        cam = gradcam_for_image(model, img[0], int(label[0]), device)
        # resize cam to 32x32
        cam_resized = np.clip(cam, 0,1)
        # overlay
        plt.figure(figsize=(4,2))
        plt.subplot(1,2,1)
        plt.imshow(img_vis)
        plt.title(f"label:{int(label)}")
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(img_vis)
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)
        plt.title('Grad-CAM')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'gradcam_{i}.png'))
        plt.close()

# Main: training loop, checkpoint, artifact saving
# Main: training loop, checkpoint, artifact saving
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--out', type=str, default='runs/cls')
    args = parser.parse_args()

    # ✅ Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.backends.cudnn.benchmark = True  # speed up on fixed-size inputs

    Path(args.out).mkdir(parents=True, exist_ok=True)

    trainloader, testloader, trainset, testset = get_dataloaders(args.batch_size)
    model = resnet18_cifar(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)

    # ✅ AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    best_acc = 0.0

    for epoch in range(1, args.epochs+1):
        # ---- Training ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in trainloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()

            # ✅ Forward with autocast
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            # ✅ Backward with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Validation ----
        val_loss, val_acc, ypred, ytrue = eval_epoch(model, testloader, criterion, device)

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch}/{args.epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # checkpoint
        torch.save({'epoch': epoch,
                    'model_state': model.state_dict(),
                    'opt': optimizer.state_dict()},
                   os.path.join(args.out, 'latest.pth'))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out, 'best_model.pth'))

    # save figures
    plot_curves(history, args.out)
    classes = trainset.classes
    save_confusion_matrix(ytrue, ypred, classes, os.path.join(args.out, 'confusion_matrix.png'))
    save_prediction_grids(model, testset, device, args.out, n=16)
    save_gradcam_images(model, testset, device, args.out, num=6)
    print("Done. Artifacts saved to", args.out)

if __name__ == '__main__':
    main()


