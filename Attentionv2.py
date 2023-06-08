import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

class SurvDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]


class Net(nn.Module):
    def __init__(self,seq_len,output_size,hidden_size,layers=3,heads=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1,hidden_size,kernel_size=3,padding=1)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size,heads) for _ in range(layers)
        ])
        self.linear1_layers = nn.ModuleList([
            nn.Linear(hidden_size,hidden_size) for _ in range(layers)
        ])
        self.linear2_layers = nn.ModuleList([
            nn.Linear(hidden_size,hidden_size) for _ in range(layers)
        ])
        self.layer_norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(layers)
        ])
        self.linear_layer3 = nn.Linear(seq_len*hidden_size,640)
        self.linear_layer4 = nn.Linear(640,2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.transpose(self.conv1(x),1,2)
        # print(x.shape)
        # x = x.squeeze(2)
        for self_attention, linear1, linear2, layer_norm in zip(
            self.attention_layers,
            self.linear1_layers,
            self.linear2_layers,
            self.layer_norm_layers
        ):
            residual = x
            x, _ = self_attention(x, x, x)
            x = residual + x
            x = layer_norm(x)
            residual = x
            x = linear2(torch.relu(linear1(x)))
            x = residual + x
            x = layer_norm(x)
            x = self.dropout(x)


        x = x.flatten(start_dim=1)
        # print(x.shape)
        x = self.linear_layer3(x)
        x = self.linear_layer4(x)
        return x

data = pd.read_csv("clean_data2.csv")
print(data.head())

label = data.iloc[:,0].values
data = data.iloc[:,2:].values

dataset = SurvDataset(data,label)
n_train = int(0.8*len(dataset))
n_test = len(dataset)-n_train
train_dataset, test_dataset = random_split(dataset,[n_train, n_test])

train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=True)

hidden_size = 128
heads = 1
output_szie = 2
layers = 3
net = Net(data.shape[1],output_szie,hidden_size,layers,heads)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.0001)

n_epoches = 20
train_losses = []
for epoch in range(n_epoches):
    net.train()
    cum_loss = 0.0
    for data,label in train_loader:
        data = torch.transpose(data.unsqueeze(-1),1,2) # b,input(1),len(489)
        optimizer.zero_grad()

        output = net(data)
        # print(label.dtype)
        loss = criterion(output,label.long())
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()

        cum_loss += loss.item()
    print(f"Epoch {epoch+1}/{n_epoches}, Loss: {cum_loss/len(train_loader)}")

    net.eval()
    with torch.no_grad():
        correct = 0
        for data,label in test_loader:
            data = torch.transpose(data.unsqueeze(-1),1,2)
            res = torch.argmax(net(data),dim=1)
            # print(res.shape)
            correct += (res == label).sum().item()
        acc = correct / n_test
        print(f"Test acc: {acc * 100} %")


"""      
Epoch 1/20, Loss: 3.533529340317755
Test acc: 60.30534351145038 %
Epoch 2/20, Loss: 1.8882056655305806
Test acc: 62.213740458015266 %
Epoch 3/20, Loss: 1.5310340383739183
Test acc: 64.12213740458014 %
Epoch 4/20, Loss: 0.7396454435180534
Test acc: 66.79389312977099 %
Epoch 5/20, Loss: 0.48603346497949323
Test acc: 68.32061068702289 %
Epoch 6/20, Loss: 0.4159606619540489
Test acc: 60.30534351145038 %
Epoch 7/20, Loss: 0.3519055145707997
Test acc: 68.70229007633588 %
Epoch 8/20, Loss: 0.33186189581950504
Test acc: 66.41221374045801 %
Epoch 9/20, Loss: 0.3203759516278903
Test acc: 66.03053435114504 %
Epoch 10/20, Loss: 0.2733090007959893
Test acc: 67.17557251908397 %
Epoch 11/20, Loss: 0.24873694060652546
Test acc: 66.03053435114504 %
Epoch 12/20, Loss: 0.25866541723636066
Test acc: 67.55725190839695 %
Epoch 13/20, Loss: 0.28425068718691665
Test acc: 58.778625954198475 %
Epoch 14/20, Loss: 0.22613087488394795
Test acc: 64.8854961832061 %
Epoch 15/20, Loss: 0.23203815847183717
Test acc: 66.79389312977099 %
Epoch 16/20, Loss: 0.22195296915191592
Test acc: 67.17557251908397 %
Epoch 17/20, Loss: 0.1591126105155457
Test acc: 66.79389312977099 %
Epoch 18/20, Loss: 0.1702212126022487
Test acc: 63.358778625954194 %
Epoch 19/20, Loss: 0.15048391834804506
Test acc: 65.2671755725191 %
Epoch 20/20, Loss: 0.1433962253486794
Test acc: 67.55725190839695 %
"""