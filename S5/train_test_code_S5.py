# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 22:12:29 2023

@author: syeda
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



# Train Phase transformations
train_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])

# Test Phase transformations
test_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])

train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)



from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from typing import Union,List
import argparse

parser= argparse.ArgumentParser(description="Digit Classifier")
parser.add_argument("-n","--norm_type",default="BN",type=str,help="Enter 'BN' for Batch norm 'LN' for layer norm and 'GN' for Group Norm")
parser.add_argument("-E","--EPOCHS",default=20,type=int)
parser.add_argument("-ng","--n_groups",default=0,type=int,help="no of groups you want for group normalization.Pass 1 if using layer norm")
parser.add_argument("-dv","--dropout_value",default=0.05,type=float,help="dropout value")
args= parser.parse_args()


from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader,test_losses, test_acc, misclassified):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for i in range(len(pred)):
              if pred[i]!= target[i]:
                misclassified.append([data[i], pred[i], target[i]])
            
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="Loss")
ax1 = fig.add_subplot(122, title="Accuracy")




def plot(epochs,**kargs):    
    ax0.plot(epochs, losses['train'], 'bo-', label='train loss')
    ax0.plot(epochs, losses['val'], 'ro-', label='val loss')
    ax1.plot(epochs, accuracy['train'], 'bo-', label='train accuracy')
    ax1.plot(epochs, accuracy['val'], 'ro-', label='val accuracy')
    if epochs[0] == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./LossGraphs', 'train.jpg'))
    
    
def plot_misclassified(misclassified):
    fig = plt.figure(figsize = (10,10))
    for i in range(25):
        sub = fig.add_subplot(5, 5, i+1)
        plt.imshow(misclassified[i][0].cpu().numpy().squeeze(),cmap='gray',interpolation='none')
        
        sub.set_title("Pred={}, Act={}".format(str(misclassified[i][1].data.cpu().numpy()),str(misclassified[i][2].data.cpu().numpy())))
    
    plt.tight_layout()
    plt.show()   
    
    
 # Load the test dataset
test_dataset = MNIST(root='./data', 
                                train=False, 
                                transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=False)

# Initialize the misclassified images array
norms = {'batch':b_model,'layer':l_model,'group':g_model}
misclassified_images = []
num_images = 20

# Predict the test set
for norm,model in norms.items():
    misclassified_images = []
    with torch.no_grad():
      for images, labels in test_loader:
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          # Append the misclassified images to the array
          for i in range(len(predicted)):
              if predicted[i] != labels[i]:
                  misclassified_images.append((images[i], predicted[i], labels[i]))
                  if len(misclassified_images) == num_images:
                      break
          if len(misclassified_images) == num_images:
              break

  # Plot the misclassified images
    fig, axs = plt.subplots(10, 2, figsize=(10, 10))
    fig.tight_layout()
    axs = axs.ravel()
    for i, (image, pred, label) in enumerate(misclassified_images):
        axs[i].imshow(image.cpu().numpy().squeeze(), cmap='gray')
        axs[i].set_title(f'Pred: {pred}, Label: {label}')
        axs[i].axis('off')

    plt.suptitle(f'Misclassified images for {norm} normalization')
    plt.show()
    
    
    

if __name__=="__main__":
  
  model=Net(args.norm_type,args.n_groups,args.dropout_value).to(device)
  optim_adam=optim.Adam(model.parameters(),lr=2.15E-02)
  scheduler= ReduceLROnPlateau(optim_adam,threshold=0.0001,patience=1,factor=.215,mode='max')
  summary(model,input_size=(1,28,28))
 
  losses, accuracy={},{}
  losses["train"]=[]
  losses["val"]=[]
  accuracy["train"]=[]
  accuracy["val"]=[]
  
  
  for epoch in range(args.EPOCHS):
    print(f"EPOCH {epoch}")
    train_epoch_loss,train_epoch_accuracy = train(train_loader,model,optim_adam)
    val_epoch_loss,val_epoch_accuracy = test(test_loader,model,optim_adam)

    scheduler.step(val_epoch_accuracy)
    losses["train"].append(train_epoch_loss)
    accuracy["train"].append(train_epoch_accuracy)

    losses["val"].append(val_epoch_loss)
    accuracy["val"].append(val_epoch_accuracy)
    
  print(len(losses["train"]))
  plot(range(args.EPOCHS),losses=losses,accuracy=accuracy) 