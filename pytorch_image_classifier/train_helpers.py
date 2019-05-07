import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
from collections import OrderedDict

import argparse

#parse arguments given by user at the command line
def parse_args():
    #define parser
    parser = argparse.ArgumentParser(description='Image Classifier Training Script Arguments')
    
    #add arguments
    #data directory
    parser.add_argument('data_dir', type=str, default='flowers', help='data directory')
    #save directory to path
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='where to save checkpoint of model')
    #model name
    parser.add_argument('--arch', type=str, default='vgg16', help='model architecture string from torchvision.models')
    #batch size
    parser.add_argument('--batch', type=int, default=32, help='batch size for dataloader iterables')
    #learning rate
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    #epochs
    parser.add_argument('--epochs', type=int, default=5, help='how many epochs to train for')
    #hidden units
    parser.add_argument('--hidden_units', type=int, default=1024, help='number of hidden units for hidden layer in classifier')
    #gpu or cpu
    parser.add_argument('--gpu', type=bool, default=True, help='True to run on gpu, False to run on cpu')
    
    #return arguments
    args = parser.parse_args()
    return args



#helper method that constructs the transforms for each of the train, val, and test sets
#currently, the transforms are static when called in train.py
def transform():
    #training transforms
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])

    #validation transforms
    val_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])

    #test transforms
    test_transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])

    
    return train_transform, val_transform, test_transform



#helper function to load in data
def load_data(train_dir, valid_dir, test_dir, train_transform, val_transform, test_transform):
    # load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(valid_dir, transform=val_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    
    return train_data, val_data, test_data



#define helper method to return dataloaders for iterating through data
def dataloader(train_data, val_data, test_data, batch, shuffle):
    #define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=shuffle)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=batch, shuffle=shuffle)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch, shuffle=shuffle)
    
    return trainloader, validloader, testloader


#allow user to be able to construct a vgg16 or  model
def construct_model(input):
    if input == 'vgg16':
        #load in vgg16
        model = models.vgg16(pretrained=True)
        #returning in_features will allow us to construct classifier,
        #regardless of model chosen
        in_features = 25088
        
    else:
        #load in resnet18
        model = models.alexnet(pretrained=True)
        in_features = 9216
        
    return model, in_features


#construct a classifier that fits with either vgg16 or alexnet
def construct_classifier(model, in_features, hidden):
    #because our model is already pretrained, we don't want to change the weights
    #turn back propagation off
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(in_features, hidden)),
    ('relu1', nn.ReLU()),
    ('drop1', nn.Dropout(0.2)),
    
    ('fc2', nn.Linear(hidden, hidden//2)),
    ('relu2', nn.ReLU()),
    ('drop2', nn.Dropout(0.2)),

    
    ('fc3', nn.Linear(hidden//2, 102)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
    
    return classifier


def finish_model(model, classifier, loss, opt, lr):
    #set the pretrained network's classifier to our newly created one
    model.classifier = classifier

    #define loss metric and optimizer
    criterion = nn.NLLLoss()

    #update parameters for classifier, not features
    if opt == 'Adam':
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)
    
    return model, criterion, optimizer
    

#allow user to train on gpu or cpu here
def gpu_or_cpu(model, dev):
    #change to gpu
    model.to(dev)
    
    

#helper method that abstracts the model training
def train(epochs, print_every, trainloader, validloader, model, device, optimizer, criterion):

    #training
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = print_every

    #lists for saving loss, will be used for plotting later
    train_losses, val_losses, acc = [], [], []

    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
        
            #convert FloatTensors to cudaTensors to run on gpu
            images, labels = images.to(device), labels.to(device)
        
            #return gradient to zero for new computation
            optimizer.zero_grad()
        
            #forward propagation, calculate log probabilities
            logps = model.forward(images)
        
            #compute loss for model, labels
            loss = criterion(logps, labels)
        
            #backprop
            loss.backward()
        
            #gradient step
            optimizer.step()
        
            #now append loss
            running_loss += loss.item()
        
            #validation step
            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
            
                with torch.no_grad():
                    for images, labels in validloader:
                    
                        images, labels = images.to(device), labels.to(device)
                    
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                        val_loss += batch_loss.item()
                    
                        probs = torch.exp(logps)
                        top_prob,top_label = probs.topk(1, dim=1)
                    
                        equality = top_label == labels.view(*top_label.shape)
                    
                        accuracy += torch.mean(equality.type(torch.cuda.FloatTensor)).item()
                
                    print("{}/{}:  ".format(epoch+1, epochs),
                        "Training_loss: {:.3f}".format(running_loss/print_every),
                        "Validation loss: {:.3f}".format(val_loss/len(validloader)),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                    #append calculated losses from each epoch
                    train_losses.append(running_loss / print_every)
                    val_losses.append(val_loss / len(validloader))
                    acc.append(accuracy / len(validloader))
                
                    #turn training back on at the end of validation loop
                    running_loss = 0
                    model.train()
                    
     #return these lists for plotting/analysis        
    return train_losses, val_losses, acc



#save model to checkpoint
def save_model(epochs, classifier, optimizer, train_data, val_data, test_data, model, mod, path):
    #save the checkpoint
    checkpoint = {'epochs': epochs,
                  'classifier': classifier,
                'optim': optimizer.state_dict,
                'train_idx': train_data.class_to_idx,
                'val_idx': val_data.class_to_idx,
                'test_idx': test_data.class_to_idx,
                'state_dict': model.classifier.state_dict(),
                 'architecture': mod
                }

    torch.save(checkpoint, path)
