import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms as t
from PIL import Image
import json


def get_cat_names(filepath):
    if filepath:
        with open(filepath, 'r') as f:
            return json.load(f)


def process_data(data_dir):
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {'train': t.Compose([t.RandomRotation(30),
                                           t.RandomResizedCrop(224),
                                           t.ToTensor(),
                                           t.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])]),
                       'valid': t.Compose([t.Resize(255),
                                           t.CenterCrop(224),
                                           t.ToTensor(),
                                           t.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])]),
                       'test': t.Compose([t.Resize(255),
                                          t.CenterCrop(224),
                                          t.ToTensor(),
                                          t.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])}

    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                      'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])}

    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64,),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64,)}

    return dataloaders, image_datasets['train'].class_to_idx


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Tensor
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    cols, rows = image.size
    if cols >= rows:
        image.thumbnail((cols, 256))
    else:
        image.thumbnail((256, rows))

    cols, rows = image.size
    new_square_size = 224
    left = (cols - new_square_size)/2
    top = (rows - new_square_size)/2
    right = (cols + new_square_size)/2
    bottom = (rows + new_square_size)/2

    image = image.crop((left, top, right, bottom))

    np_image = np.array(image)
    np_image = (np_image/255-[0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]

    return torch.from_numpy(np_image.transpose(2, 0, 1))


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
