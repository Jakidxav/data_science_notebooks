import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
from collections import OrderedDict
import json

from train_helpers import *

import argparse


#begin script
#get arguments from user
args = parse_args()

#specify data directories
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#constant declarations
batch = args.batch
shuffle = True

#enter either 'vgg16' or 'alexnet'
mod = args.arch

#change loss, optimizer, and learning rate hyperparameters here
loss = nn.NLLLoss()
#enter either 'Adam' or 'SGD'
optim = 'Adam'
learning_rate = args.lr
hidden_units = args.hidden_units

#change to 'cuda' for gpu or 'cpu' for cpu here
gpu = args.gpu

if gpu == True:
    dev = 'cuda'
else:
    dev = 'cpu'

#for training and reporting
epochs = args.epochs
print_every = 60

#define where to save model to
path = args.save_dir


#begin script
#construct transforms to perform on data
train_transform, val_transform, test_transform = transform()

#load datasets with ImageFolder
train_data, val_data, test_data = load_data(train_dir, valid_dir, test_dir,
                                            train_transform, val_transform, test_transform)

#define the dataloaders
trainloader, validloader, testloader = dataloader(train_data, val_data, test_data, batch, shuffle)

#construct the model
model, input_shape = construct_model(mod)

#define our own classifier
classifier = construct_classifier(model, input_shape, hidden_units)

#finish model setup here
model, criterion, optimizer = finish_model(model, classifier, loss, optim, learning_rate)

#change to 'cuda' for gpu or 'cpu' for cpu here
gpu_or_cpu(model, dev)

train_losses, val_losses, acc = train(epochs, print_every, trainloader, validloader, 
                                      model, dev, optimizer, criterion)

#save model output
save_model(epochs, classifier, optimizer, train_data, val_data, test_data, model, mod, path)

print('Model successfully trained and saved to checkpoint.')
