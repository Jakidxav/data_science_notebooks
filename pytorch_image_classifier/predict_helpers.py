import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
from collections import OrderedDict
import json

import matplotlib.pyplot as plt
from PIL import Image

import argparse



#parse arguments given by user at the command line
def parse_args_predict():
    #define parser
    parser = argparse.ArgumentParser(description='Image Classifier Prediction Script Arguments')
    
    #add arguments
    #path for image to predict on
    parser.add_argument('image_path', type=str, help='image path')
    #load model from checkpoint
    parser.add_argument('checkpoint', type=str, help='path of model to load')
    #number of top classes to display
    parser.add_argument('--top_k', type=int, default=5, help='number of classes to show predictions for')
    #load dictionary with category names
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='load dictionary containing category names/class indices')
    #gpu or cpu
    parser.add_argument('--gpu', type=bool, default=True, help='True to run on gpu, False to run on cpu')
    
    #return arguments
    args = parser.parse_args()
    return args



#load our saved model from its checkpoint
def load_model(filename):
    #load checkpoint
    checkpoint = torch.load(filename)
    
    #load model architecture
    model = getattr(models, checkpoint['architecture'])(pretrained=True)
    
    model.train_idx = checkpoint['train_idx']
    model.classifier = checkpoint['classifier']
    model.classifier.load_state_dict(checkpoint['state_dict'])
    
    #create dictionary of other saved keys
    hyperparameters = checkpoint
    del hyperparameters['state_dict']
    
    return hyperparameters, model



#process an image from user given image file path
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #load in image
    pil_image = Image.open(image_path)
    
    #resize
    if pil_image.size[1] > pil_image.size[0]:
        pil_image.thumbnail((256, 9999))
    else:
        pil_image.thumbnail((9999, 256))
    
    #recalculate width and height
    width = pil_image.width
    height = pil_image.height
    
    #calculate cropping points
    to_crop = 224
    
    left = int((width - to_crop) / 2)
    bottom = int((height - to_crop) / 2)
    
    right = left + 224
    top = bottom + 224
    
    #crop the image
    cropped = pil_image.crop((left, bottom, right, top))
    
    #scale color channels
    np_image = np.array(cropped)
    np_image = np.array(np_image)
    np_image = np_image / 255
    
    #normalize and standardize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image_norm = np_image - mean
    image_std = image_norm / std
    
    #reorder image dimensions
    ready_image = np.transpose(image_std, (2, 0, 1))
    
    return ready_image



#helper method to display the users image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.transpose(image, (1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    if title==None:
        ax.set_title('')
    else:
        ax.set_title(title)
    
    return ax



#helper method to display user's image and our predictions
def plot(path, title, names, top_probs):
    fig = plt.figure(figsize=(4, 8))

    ax = fig.add_subplot(2, 1, 1)

    #turn off image axis labels
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    #plot the actual image
    imshow(process_image(path), ax=ax, title=title)

    #add second figure, show probabilities with class labels
    ax2 = fig.add_subplot(2, 1 , 2)
    ax2.barh(names, top_probs)
    plt.show()
    
    

#method that actually makes predictions
#calls process_image(), returns top probabilities and classes
def predict(image_path, model, gpu, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #process image
    image = process_image(image_path)
    
    with torch.no_grad():
        #model.eval()
        model.to(device)
        model.eval()
        
        if gpu == True:
            #convert to cuda tensor, because we are working on the gpu
            image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
        else:
            #predict on cpu
            image = torch.from_numpy(image).type(torch.FloatTensor)
            
        #move image to device
        image = image.to(device)
        
        #inserts a singleton dimension in dim=0, essentially makes a tensor
        image = image.unsqueeze(0) 
        
        #calculate log probabilities
        log_probs = model.forward(image)
                
        #convert from log probabilites to probabilities
        probs = torch.exp(log_probs)
                
        #take the top class probability
        top_probs, top_classes = probs.topk(topk, dim=1)
    
        #return tensors to the cpu so that we can manipulate them
        #then return numpy array versions of those tensors
        top_probs, top_classes = top_probs[0].cpu().numpy(), top_classes[0].cpu().numpy()
        
        return top_probs, top_classes
    
    
    
