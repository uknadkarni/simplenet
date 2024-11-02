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
        # Resets the gradients of all parameters (weights and biases) in the model to zero.
        # Why it's necessary:
        #   PyTorch accumulates gradients by default. If you don't zero the gradients, they will be accumulated with every backward pass.
        #   This accumulation can lead to incorrect gradient calculations and updates.
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Updates the parameters of the model based on the computed gradients.
        # Applies the optimization algorithm (in this case, SGD - Stochastic Gradient Descent) to update the parameters.
        optimizer.step()
        
