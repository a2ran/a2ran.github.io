---
title: "Classify CIFAR-10 dataset with CNN"
toc: true
use_math: true
categories:
  - app
tags:
  - [Deep Learning, CNN, classification, projects]
date: 2023-02-05
last_modified_at: 2023-02-05
sitemap:
  changefreq: daily
  priority: 1.0
---

Classifying CIFAR-10 Dataset with over 70% test accurate Pytorch Neural Network 

## Prerequisites

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
```

## Model Architecture

<img src = '/assets/images/app/cnn/1.png'>

### Model Code
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # 30 = (32 - 3)/1 + 1
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32,  kernel_size = 3)
        # 13 = (15 - 3)/1 + 1
        self.conv2 = nn.Conv2d(32, 64, 3)
        # 11 = (13 - 3)/1 + 1
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        self.fc1 = nn.Linear(128 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p = 0.1, inplace = False)
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, (2, 2))
        
        x = x.view(-1, self.num_flat_features(x))
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features
```

## Train Model

<img src = '/assets/images/app/cnn/2.png'>

```python
# Hyperparameters

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(myNet.parameters(),
                     lr = 0.001,
                     momentum = 0.9)
```

```python
for epoch in range(10):
    
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()  # 가중치 초기화

        outputs = myNet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()   
        optimizer.step() 

        running_loss += loss.item()
        
        if i % 2000 == 1999:
            print("Epoch: {}, Batch: {}, Loss: {}".format(epoch+1, i+1, running_loss/2000))
            running_loss = 0.0
```

## Result

<img src = '/assets/images/app/cnn/3.png'>

### Test Accuracy

```python
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device) 
        
        outputs = Loaded_Net(images) # y_pred
        _, predicted = torch.max(outputs.data, axis=1) 
        total += labels.size(0) # 전체 갯수
        correct += (predicted == labels).sum().item() 
    
    print(100 * correct / total)
```

> 72.38

### Classification Accuracy

```python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = Loaded_Net(images)
        _, predicted = torch.max(outputs.data, axis=1)
        c = (predicted == labels).squeeze()
        for i in range(4): # 각각의 batch(batch-size : 4) 마다 계싼
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print("Accuracy of {}: {}%".format(class_list[i], 100 * class_correct[i] / class_total[i]))
```

> Accuracy of plane: 77.8% <br>
> Accuracy of car: 83.2% <br>
> Accuracy of bird: 69.5% <br>
> Accuracy of cat: 54.8% <br>
> Accuracy of deer: 65.9% <br>
> Accuracy of dog: 64.6% <br>
> Accuracy of frog: 69.8% <br>
> Accuracy of horse: 76.4% <br>
> Accuracy of ship: 78.9% <br>
> Accuracy of truck: 82.9%
