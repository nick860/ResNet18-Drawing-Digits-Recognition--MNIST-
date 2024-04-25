import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch 
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib
import matplotlib.pyplot as plt

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained: # if we want to use the pretrained model
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transfrom = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights) 
        else:
            self.transfrom = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat grayscale channel 3 times, to make it RGB
            ])
            self.resnet18 = resnet18() # this is the resnet model

        in_features_dim = self.resnet18.fc.in_features # get the input features of the model
        self.resnet18.fc = nn.Identity() # remove the last layer of the model
        if probing:
            for name, param in self.resnet18.named_parameters():
                param.requires_grad = False

        self.logistic_regression = nn.Linear(in_features_dim, 10) # add a logistic regression layer to the model

        
    def forward(self, x):
        features = self.resnet18(x) 
        return self.logistic_regression(features)
    
def compute_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for (imgs, labels) in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs) # squeeze from shape (batch_size, 1) to (batch_size)
            total += labels.size(0)
            correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    return correct / total , 0 #torch.cat(prediction, 0).cpu().numpy()

def get_data_loaders(transform, batch_size):
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = None
    return train_loader, val_loader, test_loader

def run_training_epoch(model, criterion, optimizer, train_loader, device):
    model.train()
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        imgs, labels = imgs.to(device), labels.to(device) # img in shape (batch_size, 3, 224, 224), labels in shape (batch_size)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze() # squeeze from shape (batch_size, 1) to (batch_size)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

def model_predict(model):
      transform = model.transfrom
      batch_size = 32
      train_loader, val_loader, test_loader = get_data_loaders(transform, batch_size) # get the data loaders
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # get the device
      model = model.to(device) # move the model to the device
      test_acc, prediction = compute_accuracy(model, test_loader, device)
      print(f'Accuracy: {test_acc}')

def train_baseline(model, num_epochs, learning_rate, batch_size):
    test_acc = 0
    transform = model.transfrom
    train_loader, val_loader, test_loader = get_data_loaders(transform, batch_size=32) # get the data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # get the device
    model = model.to(device) # move the model to the device
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001) # define the optimizer
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    for i in range(num_epochs):
        print("epoch number: ", i)
        run_training_epoch(model, criterion, optimizer, train_loader, device)
        lr_scheduler.step()
        test_acc, prediction = compute_accuracy(model, test_loader, device)
        print(f'Test accuracy: {test_acc:.4f}')
   