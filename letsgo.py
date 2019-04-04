#!/usr/bin/python

from datasets import CIFAR10Limited
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = Compose([ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    RandomCrop((220, 220)),
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomRotation(45)])

train_dataset = CIFAR10Limited('cifar_data', split='train', transform=transforms, download=True)
val_dataset = CIFAR10Limited('cifar_data', split='val', transform=transforms, download=True)
test_dataset = CIFAR10Limited('cifar_data', split='test', transform=transforms, download=True)

batch_size=32

training_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

class MLPModel(nn.Module):
    
    def __init__(self):
        super(MLPModel, self).__init__()
        self.layers = nn.Sequential(nn.Linear(32*32*3, 128),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(128, 10))
    
    def forward(self, input):
        input = input.view(input.size(0), -1)
        return self.layers(input)



def train(model):     
    model.train()
    
    train_losses = []
    val_losses = []
        
    for iteration, (images, labels) in enumerate(training_loader):
        #IS THIS NEEDED ? okay, not on my machine but on the cluster indeed
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        out = model(images)
        
        loss = loss_function(out, labels)

        train_losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
            
        if iteration % 100 == 0:
            print("Training iteration ", iteration, "out of ", len(test_loader.dataset)/batch_size, "loss = ", loss.item())
            
    epoch_train_loss = np.mean(train_losses)
    
    return(epoch_train_loss)
    
def validate(model):

    model.eval()

    val_losses = []
    
    with torch.no_grad():
        for iteration, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            out = model(images)
            
            loss = loss_function(out, labels)

            val_losses.append(loss.item())
            
            if iteration % 100 == 0:
                 print("Validation iteration ", iteration, "out of ", len(test_loader.dataset)/batch_size, "loss = ", loss.item())
            
    epoch_val_loss = np.mean(val_losses)
    
    return(epoch_val_loss)

def test(model):
    model.eval()
    
    losses = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images.to(device)
            labels.to(device)
            
            out = model(images)
            
            loss = loss_function(out, labels)
           
            losses.append(loss.item())
    
    average_loss = np.mean(losses)
    
    return(average_loss)


model = MLPModel()
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


n_epochs = 10
training_losses = []
validation_losses = []

for epoch in range(n_epochs):
    print('Epoch {}'.format(epoch+1))

    trained = train(model)
    training_losses.append(trained)

    validated = validate(model)
    validation_losses.append(validated)


final = test(model)

print("testing loss = ", final)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(np.arange(n_epochs), training_losses, color="blue", label="train loss")
plt.plot(np.arange(n_epochs), validation_losses, color="red", label="val loss")
plt.legend(loc='upper right')
plt.subplot(1,2,2)
plt.plot(np.arange(n_epochs), training_accuracies, color="blue", label="train accuracy")
plt.plot(np.arange(n_epochs), validation_accuracies, color="red", label="val accuracy")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

