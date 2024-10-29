import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleNet
from data_loading import dataloader

model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters, lr=0.01)

num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
