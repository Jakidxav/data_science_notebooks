import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
from collections import OrderedDict
import json

import matplotlib.pyplot as plt
from PIL import Image

from predict_helpers import *

import argparse


#begin script
#get arguements
args = parse_args_predict()

#data files and directories
path = args.image_path

#path for where your model is saved
checkpoint_path = args.checkpoint

#change device to 'cpu' if you want to train on cpu instead of gpu
gpu = args.gpu
if gpu == True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#convert categories to names
categories_file = args.category_names

with open(categories_file, 'r') as f:
    cat_to_name = json.load(f)

#change top_k based on how many classes you want to see
top_k = args.top_k

#now let's load our model, and then make/visualize our predictions
hyperparams, model = load_model(checkpoint_path)

#get top probabilities and classes
top_probs, top_classes = predict(path, model, gpu, device, top_k)
      
#read in class labels and indices
train_idx = hyperparams['train_idx']
        
#we need to switch indices and class class labels dictionary
#so that we can look up our top class predictions
idx_to_class = { v : k for k,v in train_idx.items()}

#now convert indices to class labels
top_classes = [idx_to_class[idx] for idx in top_classes]

#get names from top classes
names = [cat_to_name[idx] for idx in top_classes]

#get the name of the top flower predicted
idx = np.argmax(top_probs)
flower_name = cat_to_name[top_classes[idx]]

#plot image and predictions
print('Top Probabilities: {}'.format(top_probs), '\n')
print('Top Classes: {}'.format(top_classes), '\n')
print('The corresponding class labels are: {}'.format(names), '\n')
print('This application thinks your image is a <<{}>>.'.format(flower_name))

