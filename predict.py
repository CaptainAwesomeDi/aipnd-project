import argparse
import torch
from torch import nn
from collections import OrderedDict
from PIL import Image
import numpy as np
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='provide the image to predict')
    parser.add_argument('checkpoint_path', type=str, help='provide the checkpoint path for trained model')
    parser.add_argument('--gpu', action='store_true', default=False, help='enable gpu mode')
    parser.add_argument('--top_k', type=int, default=5, help='print number of top predictions')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='provide the file to map real name')
    return parser.parse_args()


def process_image(image):
    size = 256, 256
    im = Image.open(image)
    im.thumbnail(size)

    width, height = im.size
    new_width, new_height = 224, 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    im = im.crop((left, top, right, bottom))

    np_image = np.array(im)
    np_image = np_image / 255
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    np_image = (np_image - mean) / std
    image = np.transpose(np_image, (2, 0, 1))

    return torch.from_numpy(image)


def load_checkpoint_and_build_model(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['arch']
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    d = checkpoint['dropout']
    hidden_layers = checkpoint['hidden_layers']

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_layers[0])),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=d)),
        ('fc2', nn.Linear(hidden_layers[0], hidden_layers[1])),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=d)),
        ('fc3', nn.Linear(hidden_layers[1], output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model


def predict(image_path, model, topk, device):
    model.eval()
    model.double()
    model = model.to(device)
    image = process_image(image_path)
    image = image.to(device).unsqueeze(0)
    with torch.no_grad():
        output = model.forward(image)
        ps, ps_labels = torch.topk(output, topk)
        ps = ps.exp()
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    top_class = []
    for label in ps_labels.cpu().numpy()[0]:
        top_class.append(idx_to_class[label])
    return ps.cpu().numpy()[0], top_class


def main():
    arguments = get_args()
    image_path = arguments.image_path
    checkpoint_path = arguments.checkpoint_path
    gpu = arguments.gpu
    top_k = arguments.top_k
    category_names = arguments.category_names
    print('=====LOADED SETTINGS=====')
    print('Image path: {}'.format(image_path))
    print('Checkpoint file path: {}'.format(checkpoint_path))
    print('GPU mode: {}'.format(gpu))
    print('# of top predictions: {}'.format(top_k))
    print('Category names file path: {}'.format(category_names))

    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = load_checkpoint_and_build_model(checkpoint_path)
    print('Checkpoint file loaded successfully')

    print('Starting predicting...')
    ps, tclass = predict(image_path, model, top_k, device)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    print('===== Predicted Flower =====')
    print('Flower: {}'.format(cat_to_name[tclass[0]]))
    print('Class: {}'.format(tclass[0]))
    print('Probability: {:.2f}%'.format(ps[0] * 100))
    print('===== Top Predictions =====')
    for idx, i in enumerate(tclass):
        print('Flower: {flower_name}'.format(flower_name=cat_to_name[tclass[idx]]))
        print('Class: {flower_class}'.format(flower_class= tclass[idx]))
        print('Probability: {flower_ps:.2f}%'.format(flower_ps=ps[idx] * 100))
        print('======================')




if __name__ == "__main__":
    main()
