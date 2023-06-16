import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

file = "Data/clean_data_final.csv"

def read_file(file):
    data = pd.read_csv(file)
    train_data = data.sample(frac=0.8,random_state=0,axis=0)
    test_data = data[~data.index.isin(train_data.index)]
    return train_data.values, test_data.values

batch_size = 8

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        feature = self.data[index][1:]
        feature = torch.FloatTensor(feature)
        label = self.data[index][0]
        label = torch.tensor(label)
        return feature, label
    def __len__(self):
        return len(self.data)

train_data, test_data = read_file(file)

train_set = MyDataset(train_data)
test_set = MyDataset(test_data)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(669, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


model = Net()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def train(epoch):
    avg_loss = 0.0
    for idx, data in enumerate(train_loader, 0):
        X, y = data
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y.long())
        avg_loss += loss
        loss.backward()
        optimizer.step()

        if idx % 10 == 9:
            print("Epoch: %d, Batch_index: %d, Loss: %.3lf"%(epoch + 1, idx + 1, avg_loss / 100))
            avg_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader, 0):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)             
            _, predicted = torch.max(y_pred.data, dim = 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print("Accuracy on test set:%d%%[%d/%d]"%(100 * correct / total, correct, total))

if __name__ == '__main__':
    epochs = 10
    for epoch in range(0, epochs):          
        train(epoch)
        test()              
    