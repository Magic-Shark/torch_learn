import wandb


# 初始化 W&B
wandb.login(key='')
wandb.init(project="mnist-example")

# 这里省略神经网络和数据加载的定义，假设你已经有一个模型、数据加载器等
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 数据加载
# 使用 torchvision 提供的 MNIST 数据集，并应用标准化处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层 28x28，输出层128个神经元
        self.fc2 = nn.Linear(128, 64)       # 隐藏层 64个神经元
        self.fc3 = nn.Linear(64, 10)        # 输出层 10个神经元，对应数字0-9

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平图像
        x = F.relu(self.fc1(x))  # ReLU激活函数
        x = F.relu(self.fc2(x))  # ReLU激活函数
        x = self.fc3(x)          # 输出层
        return x

# 3. 定义训练和评估函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  # 梯度清零
        output = model(data)   # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数

        running_loss += loss.item()

        _, predicted = output.max(1)  # 获取最大值的索引
        total += target.size(0)      # 总样本数
        correct += predicted.eq(target).sum().item()  # 计算正确的样本数

    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    # 在 W&B 上记录训练过程中的损失和准确率
    wandb.log({"train_loss": avg_loss, "train_accuracy": accuracy})

    return avg_loss, accuracy

def test(model, test_loader, criterion, device):
    model.eval()  # 设置为评估模式
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    # 在 W&B 上记录测试过程中的损失和准确率
    wandb.log({"test_loss": avg_loss, "test_accuracy": accuracy})

    return avg_loss, accuracy

# 4. 设置训练设备 (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. 实例化模型、损失函数和优化器
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器



# 6. 训练过程
epochs = 5
for epoch in range(epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_accuracy = test(model, test_loader, criterion, device)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%\n")

