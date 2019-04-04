#!/usr/bin/python

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, Grayscale, ColorJitter
from torch.utils.data import DataLoader
import torch
import torchvision
import torch.nn as nn
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--dropout', default=1e-4, type=float)
parser.add_argument('--hidden_width1', default=128, type=int)
parser.add_argument('--hidden_width2', default=128, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = Compose([RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomCrop(32, padding=4),
                    RandomRotation(45),
                    ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
                    #Grayscale()])

train_dataset = datasets.CIFAR10(root='/var/tmp/stu04', train=True, transform=transforms, download=True)
second_dataset = datasets.CIFAR10(root='/var/tmp/stu04', train=False, transform=transforms, download=True)
#test_dataset = datasets.CIFAR10('cifar_data', train=False, transform=transforms, download=True)   

val_size = int(0.5 * len(second_dataset))
test_size = len(second_dataset) - val_size
val_dataset, test_dataset = torch.utils.data.random_split(second_dataset, [val_size, test_size])

batch_size=args.batch_size

training_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


def train(model):     
    model.train()
    
    train_losses = []
    averages = 0
    val_losses = []

    for iteration, (images, labels) in enumerate(training_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        out = model(images)
        
        loss = loss_function(out, labels)

        averages += (np.argmax(out.detach(), axis=1).cuda() == labels).sum().item()
        train_losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
            
        if iteration % 100 == 0:
            print("Training iteration ", iteration, "out of ", len(test_loader.dataset)/batch_size,
                  "loss = ", loss.item(), "accuracy = ", 100*averages/((iteration+1)*batch_size), "%")
            
    accuracy = averages/len(training_loader.dataset)         
    epoch_train_loss = np.mean(train_losses)
    
    return((epoch_train_loss, accuracy))
    
def validate(model):
    model.eval()

    val_losses = []
    averages = 0
    
    with torch.no_grad():
        for iteration, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            out = model(images)
            
            loss = loss_function(out, labels)

            averages += (np.argmax(out.detach(), axis=1).cuda() == labels).sum().item()
            val_losses.append(loss.item())
            
            if iteration % 100 == 0:
                 print("Validation iteration ", iteration, "out of ", len(test_loader.dataset)/batch_size,
                  "loss = ", loss.item(), "accuracy = ", 100*averages/((iteration+1)*batch_size), "%")
            
    accuracy = averages/len(training_loader.dataset)
    epoch_val_loss = np.mean(val_losses)
    
    return((epoch_val_loss, accuracy))

def test(model):
    model.eval()
    
    losses = []
    averages = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images.to(device)
            labels.to(device)
            
            out = model(images.cuda())
            
            loss = loss_function(out.cuda(), labels.cuda())
           
            averages += (np.argmax(out.detach(), axis=1).cuda() == labels.cuda()).sum().item()
            losses.append(loss.item())
    
    accuracy = averages/len(training_loader.dataset)
    average_loss = np.mean(losses)
    
    return((average_loss, accuracy))

class MLPModel2(nn.Module):
    
    def __init__(self):
        super(MLPModel2, self).__init__()
        self.layers = nn.Sequential(nn.Linear(32*32*3, args.hidden_width1),
                                    nn.ReLU(),
                                    nn.Dropout(args.dropout),
                                    nn.Linear(args.hidden_width1,args.hidden_width2),
                                    nn.ReLU(),
                                    nn.Dropout(args.dropout),
                                    nn.Linear(args.hidden_width2, 10))
    
    def forward(self, input):
        input = input.view(input.size(0), -1)
        return self.layers(input)


model = MLPModel2()
model = model.to(device)
loss_function = nn.CrossEntropyLoss()
learning_rate = args.learning_rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

n_epochs = 25
training_losses = []
training_accuracies = []
validation_losses = []
validation_accuracies = []

for epoch in range(n_epochs):
    print('Epoch {}'.format(epoch+1))
    trained = train(model)
    training_losses.append(trained[0])
    training_accuracies.append(trained[1])
    validated = validate(model)
    validation_losses.append(validated[0])
    validation_accuracies.append(validated[1])

final = test(model)

print("final loss = ", final[0], "final accuracy = ", 100*final[1], "%")

f= open("summary.txt","a+")
print("final accuracy : ", 100*final[1], "%\t", "batch_size : ", args.batch_size, "\tlearning_rate : ", args.learning_rate, "\tweight_decay : ", args.weight_decay, "\tdropout :", args.dropout, "\thidden_with1 : ", args.hidden_width1, "\thidden_width2 : ", args.hidden_width2, file=f)
f.close()
