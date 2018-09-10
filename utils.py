import os
import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import json


def initTrainParse():
    parser = argparse.ArgumentParser(description='transfer learning')
    parser.add_argument('data_dir', help='dataset path')
    parser.add_argument('--save_dir', dest='save_dir',
                        help='checkpoint save dir')
    parser.add_argument('--arch', dest='arch',
                        help='choose your network', default='vgg19')
    parser.add_argument('--learning_rate', dest='learning_rate',
                        help='set learning_rate', default='0.001', type=float)
    parser.add_argument('--hidden_units', dest='hidden_units',
                        help='set hidden_units', default=4096, type=int)
    parser.add_argument('--epochs', dest='epochs',
                        help='set epochs', default=2, type=int)
    parser.add_argument('--gpu', action='store_true', help='set gpu')
    args = parser.parse_args()
    return args


def initPredictParse():
    parser = argparse.ArgumentParser(description='transfer learning--predict')
    parser.add_argument('input', help='image path')
    parser.add_argument('checkpoint', help='checkpoint path')
    parser.add_argument('--top_k', dest='top_k',
                        help='top_k count', default=5, type=int)
    parser.add_argument('--category_names', dest='category_names',
                        help='json file path')
    parser.add_argument('--gpu', action='store_true', help='set gpu')
    args = parser.parse_args()
    return args


def getDatasets(dirpath):
    data_transforms = {
        'train': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
        'valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
        'test': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    }
    return {
        'train': datasets.ImageFolder(dirpath + '/train/', transform=data_transforms['train']),
        'valid': datasets.ImageFolder(dirpath + '/valid/', transform=data_transforms['valid']),
        'test': datasets.ImageFolder(dirpath + '/test/', transform=data_transforms['test'])
    }


def getDataLoaders(datasets):
    return {
        'train': torch.utils.data.DataLoader(datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(datasets['test'], batch_size=32)
    }


def getModelsByArch(arch):
    if arch == 'vgg11':
        return models.vgg11(pretrained=True)
    if arch == 'vgg13':
        return models.vgg13(pretrained=True)
    if arch == 'vgg16':
        return models.vgg16(pretrained=True)
    if arch == 'vgg19':
        return models.vgg19(pretrained=True)
    if arch == 'vgg11_bn':
        return models.vgg11_bn(pretrained=True)
    if arch == 'vgg13_bn':
        return models.vgg13_bn(pretrained=True)
    if arch == 'vgg16_bn':
        return models.vgg16_bn(pretrained=True)
    if arch == 'vgg19_bn':
        return models.vgg19_bn(pretrained=True)
    return models.vgg19(pretrained=True)


def ininModelArgsAndClassifier(model, hidden_units):
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    return model


def isCudaAvaliable(gpu):
    if gpu & torch.cuda.is_available():
        return True
    return False


def loadCheckPoint(filepath):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)

        model = getModelsByArch(checkpoint['arch'])
        model = ininModelArgsAndClassifier(model, checkpoint['hidden_units'])
        optimizer = optim.Adam(model.classifier.parameters(),
                               lr=checkpoint['learning_rate'])

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.idx_to_class = {v: k for k,
                              v in checkpoint['class_to_idx'].items()}
        print('load success')
        return model, optimizer
    else:
        print('checkpoint is not exist')


def processImage(image_path):
    if os.path.isfile(image_path) != True:
        print('image is not exist')
        return

    im = Image.open(image_path)
    im = im.resize((256, 256))
    im = im.crop((16, 16, 240, 240))

    np_image = np.array(im) / 256

    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])

    for i in range(3):
        np_image[..., i] -= mean[..., i]
        np_image[..., i] /= std[..., i]

    np_image = np_image.transpose((2, 0, 1))

    return np_image


def getJSONFile(filepath):
    with open(filepath, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name
