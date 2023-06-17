import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc 

# 读取数据
data = pd.read_csv("../clean_data_final.csv")

# 将数据集分为训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=2)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.x = data.iloc[:, 1:].values
        self.y = data.iloc[:, 0].values
        self.mean = self.x.mean(axis=0)
        self.std = self.x.std(axis=0)
        self.x = (self.x - self.mean) / (self.std+1e-8)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]).float(), torch.tensor(self.y[index]).float()

train_set = CustomDataset(train_data)
test_set = CustomDataset(test_data)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

class FCNN(torch.nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = torch.nn.Linear(669, 2048) #所有列
        self.fc2 = torch.nn.Linear(2048, 1024)
        self.fc3 = torch.nn.Linear(1024, 256)
        self.fc4 = torch.nn.Linear(256, 1)
        
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 669) 
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

model = FCNN()

criterion = nn.BCELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# weight_decay = 0.001
# optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=weight_decay)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练
for epoch in range(10):
    running_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        output = model(x)
        output = output.reshape(-1)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(y)
    print('epoch: %d, loss: %.3f' % (epoch + 1, running_loss / len(train_set)))
    # 在测试集上进行验证
    total_loss = 0
    total_acc = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            output = output.reshape(-1)
            total_loss += criterion(output, y).item() * len(y)
            total_acc += ((output > 0.5).float() == y).float().sum().item() / len(y)
    avg_loss = total_loss / len(test_set)
    avg_acc = total_acc / len(test_loader)
    print('test_loss: %.3f, test_accuracy: %.3f' % (avg_loss, avg_acc))
    
# 评估模型
batch_idx = 32
model.eval()
with torch.no_grad():
    y_pred = []
    y_true = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        y_pred.extend(output.view(-1).tolist())
        y_true.extend(target.view(-1).tolist())
        
# 计算ROC曲线数据
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
# plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, color='red',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC-ROC figure')
plt.legend(loc="lower right")
plt.show()