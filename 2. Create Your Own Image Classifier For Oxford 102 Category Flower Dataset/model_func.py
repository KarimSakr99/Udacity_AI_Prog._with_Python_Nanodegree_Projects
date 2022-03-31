import torch
from torchvision import models
from torch import nn, optim


def build(arch, hidden_units, learning_rate):

    if arch == 'densenet':
        model = models.densenet121(pretrained=True)
        classifier_input = model.classifier.in_features
    else:
        model = models.mobilenet_v2(pretrained=True)
        classifier_input = model.classifier[1].in_features

    for param in model.parameters():
        param.requires_grad = False

    classifier_layers = build_classifier_layers(classifier_input, hidden_units)
    model.classifier = classifier_layers

    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    return model, optimizer, criterion


def build_classifier_layers(n_inputs, n_hidden):
    if n_hidden:
        return nn.Sequential(nn.Linear(n_inputs, n_hidden),
                             nn.ReLU(),
                             nn.Dropout(p=0.2),
                             nn.Linear(n_hidden, 102),
                             nn.LogSoftmax(dim=1))

    else:
        return nn.Sequential(nn.Dropout(p=0.2),
                             nn.Linear(n_inputs, 102),
                             nn.LogSoftmax(dim=1))


def load_build_from_checkpoint(path):
    checkpoint = torch.load(path)

    model = models.densenet121(
        pretrained=True) if checkpoint.get('arch') == 'densenet' else models.mobilenet_v2(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    criterion = nn.NLLLoss()

    return model, optimizer, criterion
