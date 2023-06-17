import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from sklearn.metrics import roc_curve, auc
import numpy as np
from torchsummary import summary


class SurvDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.Tensor(data)
        self.label = torch.Tensor(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]


"""
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
        self.linear_layer4 = nn.Linear(640,output_size)
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
"""


class Net(nn.Module):
    def __init__(self, seq_len, output_size, hidden_size, heads=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_size, kernel_size=3, padding=1)
        self.attention_layer = nn.MultiheadAttention(hidden_size, heads)
        self.linear1_layer = nn.Linear(hidden_size, hidden_size)
        self.layer_norm_layer = nn.LayerNorm(hidden_size)
        self.linear2_layer = nn.Linear(seq_len * hidden_size, 640)
        self.linear3_layer = nn.Linear(640, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.transpose(self.conv1(x), 1, 2)
        # print(x.shape)
        # x = x.squeeze(2)
        residual = x
        x, _ = self.attention_layer(x, x, x)
        x = x + residual
        x = self.layer_norm_layer(x)
        residual = x
        x = torch.relu(self.linear1_layer(x))
        x = x + residual
        x = self.layer_norm_layer(x)
        x = self.dropout(x)

        x = x.flatten(start_dim=1)
        # print(x.shape)
        x = self.linear2_layer(x)
        x = self.linear3_layer(x)
        return x


if __name__ == "__main__":
    data = pd.read_csv("clean_data_final.csv")
    print(data.head())

    label = data.iloc[:, 0].values
    data = data.iloc[:, 1:].values

    dataset = SurvDataset(data, label)
    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    torch.manual_seed(11)
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test], )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    hidden_size = 64
    heads = 1
    output_size = 2
    layers = 1

    # net = Net(data.shape[1],output_size,hidden_size,layers,heads)
    net = Net(data.shape[1], output_size, hidden_size, heads)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    train_losses = []
    test_acc = []
    n_epoches = 5
    for epoch in range(n_epoches):
        net.train()
        cum_loss = 0.0
        for data, label in train_loader:
            data = torch.transpose(data.unsqueeze(-1), 1, 2)  # b,input(1),len(489)
            optimizer.zero_grad()

            output = net(data)
            # print(label.dtype)
            loss = criterion(output, label.long())
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
        print(f"Epoch {epoch + 1}/{n_epoches}, Loss: {cum_loss / len(train_loader)}")

        net.eval()
        with torch.no_grad():
            correct = 0
            for data, label in test_loader:
                data = torch.transpose(data.unsqueeze(-1), 1, 2)
                res = torch.argmax(net(data), dim=1)
                # print(res.shape)
                correct += (res == label).sum().item()
            acc = correct / n_test
            test_acc.append(acc * 100)
            print(f"Test acc: {acc * 100} %")

    from matplotlib import pyplot as plt

    x = [i + 1 for i in range(len(train_losses))]
    y = train_losses
    plt.plot(x, y, label='y')
    plt.xticks(rotation=45)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()
    plt.show()

    x = [i + 1 for i in range(len(test_acc))]
    y = test_acc
    plt.plot(x, y, label='y')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('Test Acc')
    plt.legend()
    plt.xticks(np.arange(1, len(test_acc) + 1, 1))
    plt.show()

    net.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            data = torch.transpose(data.unsqueeze(-1), 1, 2)
            output = net(data)
            probabilities = torch.softmax(output, dim=1)
            predictions.extend(probabilities[:, 1].tolist())
            true_labels.extend(label.tolist())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    fpr, tpr, thresholds = roc_curve(true_labels, predictions)

    auc_value = auc(fpr, tpr)

    plt.plot(fpr, tpr, color="red", lw=2, label='ROC curve (AUC = {:.2f})'.format(auc_value))
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC figure')
    plt.legend(loc='lower right')
    plt.savefig("ROC.png")
    plt.show()

    print(f"AUC Value : {auc_value}")

    # model = Net(seq_len=669,output_size=2,hidden_size=64,layers=1,heads=1)
    model = Net(seq_len=669, output_size=2, hidden_size=64, heads=1)
    summary(model, input_size=(1, 669), batch_size=16)
