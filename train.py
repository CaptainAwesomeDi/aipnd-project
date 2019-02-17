import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        default='assets/flowers/', help='path to image folder')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='choose a prebuilt model')
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default=os.getcwd(),
                        help='directory to save checkpoints')
    return parser.parse_args()


def validation(model, loader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


def main():
    arguments = get_args()
    data_dir = arguments.data_dir
    arch = arguments.arch
    gpu = arguments.gpu
    save_dir = arguments.save_dir

    if gpu and torch.cuda.is_avaliable():
        device = 'cuda'
    else:
        device = 'cpu'


# Seting the folder directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define Transform
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(
    train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(
    test_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(
    valid_data, batch_size=32, shuffle=True)

#


if __name__ == "__main__":
    main()
