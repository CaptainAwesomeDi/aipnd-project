import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import os
import json
from collections import OrderedDict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        default='assets/flowers/', help='path to image folder')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='choose a prebuilt model')
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--save_dir', type=str, default=os.getcwd(),
                        help='directory to save checkpoints')
    parser.add_argument('--hidden_units', nargs=2, type=int, default=[4096, 2048],
                        help='number of hidden units list of 2 integers')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='set learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='set number of epochs to train for')
    return parser.parse_args()


def validation(model, loader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


def train(model, epochs, device, criterion, optimizer, trainloader, validloader):
    print_every = 40
    steps = 0
    model.to(device)
    running_loss = 0
    for e in range(epochs):
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)

                print("Epoch: {}/{}.. ".format(e + 1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss / len(validloader)),
                      "Test Accuracy: {:.3f}".format(accuracy / len(validloader)))

                running_loss = 0
                model.train()


def main():
    arguments = get_args()
    data_dir = arguments.data_dir
    arch = arguments.arch
    gpu = arguments.gpu
    save_dir = arguments.save_dir
    epochs = arguments.epochs
    learning_rate = arguments.learning_rate
    hidden_units = arguments.hidden_units

    print('====== SETTINGs ===== ')
    print('Data dir: {}'.format(data_dir))
    print('Arch: {}'.format(arch))
    print('Learning rate: {}'.format(learning_rate))
    print('Hidden units: {}'.format(hidden_units))
    print('Epochs: {}'.format(epochs))
    print('GPU mode: {}'.format(gpu))
    print('Checkpoint dir: {}'.format(save_dir))

    if gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Seting the folder directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    # Define Transform
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
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
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)

    # label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # choose pretrained model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    else:
        print('invalid choice please choose between alexnet and vgg16')

    output_size = 102
    for param in model.parameters():
        param.require_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units[0])),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.5)),
        ('fc3', nn.Linear(hidden_units[1], output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    print('Starting Training Now')
    train(model, epochs, device, criterion, optimizer, trainloader, validloader)
    checkpoint = {
        "input_size": input_size,
        "output_size": output_size,
        "epochs": epochs,
        "dropout": 0.5,
        "hidden_layers": hidden_units,
        "optimizer_state_dict": optimizer.state_dict,
        "class_to_idx": train_data.class_to_idx,
        "state_dict": model.state_dict()}
    torch.save(checkpoint, save_dir + '/' + 'new_checkpoint.pth')
    print('Training complete checkpoint save to path:{}'.format(save_dir+'/'+'new_checkpoint.pth'))

if __name__ == "__main__":
    main()
