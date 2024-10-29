import torch
import torch.nn as nn
# The SimpleNet neural network defined in the code has two layers in terms of fully connected (dense) layers:
# First Fully Connected Layer (fc1): This layer takes an input of size 10 and outputs a size of 5.
# Second Fully Connected Layer (fc2): This layer takes the output from the first layer (size 5) and produces a final output of size 1.
# Breakdown of Layers
## Layer 1: fc1 (Linear layer)
### Input size: 10
### Output size: 5
## Layer 2: fc2 (Linear layer)
### Input size: 5
### Output size: 1
## Activation Function
## While the ReLU activation function is applied after the first layer, it is not counted as a separate layer in the context of neural network architecture. Therefore, the total count of layers in this neural network is 2.

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the network
model = SimpleNet()
