import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO:解释参数含义，在?处填入合适的参数
batch_size = 64   #分批处理，每一批的数据量大小。 为什么选64呢？？？
learning_rate = 0.0001  #之前选取过0.01的学习率，但是模型效果不好，0.001刚刚好。
num_epochs = 30     #循环的次数
 
transform = transforms.Compose([
    transforms.ToTensor()
])
 
# root可以换为你自己的路径
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
 
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


class Network(nn.Module):
    #网络设计：（卷积+激活+池化）*2  平铺  线性层+激活   分类器
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)   #彩色照片原本有3个通道，选用3*3的卷积核，保证图片大小不变要让padding=1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)   #经过一个线性层。若不知64*8*8 ，可以先不写这个，在下面运行的时候用shape显示出来
        self.fc2 = nn.Linear(512, 256)       #最后的分类器
        self.fc3 = nn.Linear(256, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  #最大池化层，采用了2*2的卷积，步长为2？？？？？？？、为什么
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Network().to(device)  # Move model to device

criterion = nn.CrossEntropyLoss()  # 损失函数选用交叉觞损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
 
            optimizer.zero_grad()
 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
 
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
       
        train_losses.append(loss.item())    #记录训练损失
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

#测试只测试了一次
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

if __name__ == "__main__":
    train_losses = []
    train()
    test()
    print(train_losses)



plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()