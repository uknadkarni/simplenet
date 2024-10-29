import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels) -> None:
        super().__init__()
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

num_samples = 100
num_features = 10

data = torch.rand((num_samples, num_features))
labels = torch.rand(num_samples, 1)

# Create a DataLoader
dataset = MyDataset(data, labels)
dataloader = DataLoader(data, batch_size=32, shuffle=True)

for batch_data, batch_labels in dataloader:
    print("Batch Data: ", batch_data)
    print("Batch Labels: ", batch_labels)
