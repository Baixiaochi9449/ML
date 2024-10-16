import torch
import pandas as pd
import torch.nn  as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 加载波士顿房价数据集。
dataset = pd.read_csv('C:/Users/Lenovo/Desktop/ML/pytorch/Boston_House/data.csv')
X_dataset = dataset.iloc[:, :-1]  # 选取所有列，除了最后一列（特征）
y_dataset = dataset.iloc[:, -1]   # 选取最后一列（标签）
# 首先划分训练集和测试集，比例为80/20
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size=0.2, random_state=42)
# 再对训练集划分，将80%的训练集划分为新的训练集和验证集，90%的训练集用于训练，10%用于验证
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# 打印各个集合的大小以确认划分结果
print(f'Training set size: {X_train.shape[0]}')
print(f'Validation set size: {X_val.shape[0]}')
print(f'Test set size: {X_test.shape[0]}')


#处理缺失值,使用训练集的均值填充缺失值
X_train.fillna(X_train.mean(), inplace=True)   
X_test.fillna(X_train.mean(), inplace=True)   
X_val.fillna(X_train.mean(), inplace=True)     # 用训练集的均值填充验证集的缺失值
# 使用 Z-score 方法来检测并移除异常值
z_scores_train = np.abs(stats.zscore(X_train))  # 计算训练集的 Z-score
X_train_clean = X_train[(z_scores_train < 3).all(axis=1)]  # 移除 Z-score 大于 3 的异常值
y_train_clean = y_train[X_train_clean.index]  # 同时移除对应的标签

z_scores_val = np.abs(stats.zscore(X_val))  # 计算训练集的 Z-score
X_val_clean = X_val[(z_scores_val < 3).all(axis=1)]  # 移除 Z-score 大于 3 的异常值
y_val_clean = y_val[X_val_clean.index]  # 同时移除对应的标签

z_scores_test = np.abs(stats.zscore(X_test))
X_test_clean = X_test[(z_scores_test < 3).all(axis=1)]
y_test_clean = y_test[X_test_clean.index]

#数据标准化，只对x进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)  # 训练集标准化
X_test_scaled = scaler.transform(X_test_clean)        # 测试集使用相同的标准化参数
X_val_scaled = scaler.transform(X_val_clean)

# 最终结果转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_clean.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_clean.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_clean.values, dtype=torch.float32).view(-1, 1)

print(X_train_tensor.shape, y_train_tensor.shape)
print(X_test_tensor.shape, y_test_tensor.shape)
print(X_val_tensor.shape, y_val_tensor.shape)



# 神经网络模型
class BostonModel(nn.Module):
    def __init__(self):
        super(BostonModel, self).__init__()
        self.hidden1 = nn.Linear(14, 64)  # 。
        self.hidden2 = nn.Linear(64, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))  # 隐藏层1 + ReLU
        x = self.relu(self.hidden2(x))  # 隐藏层2 + ReLU
        x = self.relu(self.hidden3(x))
        x = self.output(x)              # 输出层
        return x

# 初始化模型
model = BostonModel()
# 损失函数：均方误差
criterion = nn.SmoothL1Loss()
# 使用 SGD 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 打印模型结构
print(model)



# 初始化
num_epochs = 1000
train_losses = []
val_losses = []

# Early stopping 参数
early_stopping_patience = 20  # 如果验证集损失在20个epoch内没有改善，则停止训练
best_val_loss = float('inf')    #保存损失最小的模型
epochs_no_improve = 0

for epoch in range(num_epochs):
   
    model.train() # 训练模式
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 记录训练损失
    train_losses.append(loss.item())

   
    model.eval()   # 验证模式
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())
        
    # 早停机制
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        epochs_no_improve = 0  # 重置计数器
        best_model = model.state_dict()  # 保存当前最佳模型的参数
    else:
        epochs_no_improve += 1
    

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    # 检查是否需要早停
    if epochs_no_improve == early_stopping_patience:
        print(f'Early stopping triggered at epoch {epoch+1}')
        break  # 提前终止训练

# 使用最优模型进行测试
model.load_state_dict(best_model)
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')




plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

