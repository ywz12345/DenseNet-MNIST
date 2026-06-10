import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import gzip
import os
import numpy as np
import warnings
# 屏蔽无关警告
warnings.filterwarnings("ignore")

# 设备选择
device = torch.device("cpu")
print("使用设备: cpu")

# ---------------------- 1. 自定义读取本地MNIST的数据集类 ----------------------
class LocalMNIST(Dataset):
    def __init__(self, root="./data/MNIST/raw", train=True, transform=None):
        self.transform = transform
        if train:
            images_path = os.path.join(root, "train-images-idx3-ubyte.gz")
            labels_path = os.path.join(root, "train-labels-idx1-ubyte.gz")
        else:
            images_path = os.path.join(root, "t10k-images-idx3-ubyte.gz")
            labels_path = os.path.join(root, "t10k-labels-idx1-ubyte.gz")

        # 读取并解压图片
        with gzip.open(images_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            self.images = data.reshape(-1, 28, 28).copy()  # 转为可写数组
        # 读取并解压标签
        with gzip.open(labels_path, 'rb') as f:
            self.labels = np.frombuffer(f.read(), np.uint8, offset=8).copy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------------------- 2. DenseNet 核心模块 ----------------------
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate,
                          kernel_size=3, padding=1, bias=False)
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compression=0.5):
        super().__init__()
        out_channels = int(in_channels * compression)
        self.trans = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.trans(x)

# ---------------------- 3. 简易DenseNet网络 ----------------------
class SimpleDenseNet(nn.Module):
    def __init__(self, growth_rate=12, num_classes=10):
        super().__init__()
        self.init_conv = nn.Conv2d(1, 24, 3, padding=1, bias=False)
        channels = 24

        self.block1 = DenseBlock(channels, growth_rate, 4)
        channels += 4 * growth_rate
        self.trans1 = TransitionLayer(channels)
        channels //= 2

        self.block2 = DenseBlock(channels, growth_rate, 4)
        channels += 4 * growth_rate
        self.trans2 = TransitionLayer(channels)
        channels //= 2

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.relu(self.bn(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ---------------------- 4. 数据预处理 ----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------------------- 5. 加载真实MNIST数据集 ----------------------
print("加载本地真实MNIST数据集...")
train_dataset = LocalMNIST(train=True, transform=transform)
test_dataset = LocalMNIST(train=False, transform=transform)
print("数据集加载完成！")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------------- 6. 模型、损失、优化器 ----------------------
model = SimpleDenseNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------- 7. 训练过程（5轮） ----------------------
epochs = 5
train_loss_list, train_acc_list = [], []
test_loss_list, test_acc_list = [], []

print("\n===== 开始训练 DenseNet（真实MNIST数据） =====")
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    correct = total = 0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = torch.max(output, 1)
        total += label.size(0)
        correct += (pred == label).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(train_acc)

    # 测试阶段
    model.eval()
    test_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item()
            _, pred = torch.max(output, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * correct / total
    test_loss_list.append(avg_test_loss)
    test_acc_list.append(test_acc)

    print(f"Epoch [{epoch+1}/{epochs}] | 训练损失:{avg_train_loss:.4f} 训练准确率:{train_acc:.2f}% | 测试损失:{avg_test_loss:.4f} 测试准确率:{test_acc:.2f}%")

# ---------------------- 8. 绘制训练曲线 ----------------------
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='训练损失')
plt.plot(test_loss_list, label='测试损失')
plt.title('损失变化曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='训练准确率')
plt.plot(test_acc_list, label='测试准确率')
plt.title('准确率变化曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('train_result.png')
plt.show()
print("\n训练完成，曲线已保存为 train_result.png")