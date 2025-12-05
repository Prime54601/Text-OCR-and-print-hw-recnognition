import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

from dataLoader import *

class CNNModel(nn.Module):
    """
    构建CNN模型用于识别印刷文字和手写文字
    """
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.pool = nn.MaxPool2d((2, 2))
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        
        # 第三个卷积层
        self.conv3 = nn.Conv2d(64, 64, (3, 3))
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.dropout = nn.Dropout(0.5)
        
        # 输出层（二分类：印刷文字/手写文字）
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 第一个卷积层
        x = self.pool(torch.relu(self.conv1(x)))
        
        # 第二个卷积层
        x = self.pool(torch.relu(self.conv2(x)))
        
        # 第三个卷积层
        x = torch.relu(self.conv3(x))
        
        # 展平层
        x = x.view(-1, 64 * 3 * 3)
        
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 输出层
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

class TextDataset(Dataset):
    """
    自定义数据集类
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_data(data_dir):
    """
    加载数据集
    假设数据结构为:
    data_dir/
        printed/
            *.jpg or *.png
        handwritten/
            *.jpg or *.png
    """
    images = []
    labels = []
    
    # 加载印刷文字图片
    printed_dir = os.path.join(data_dir, 'printed')
    if os.path.exists(printed_dir):
        for filename in os.listdir(printed_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(printed_dir, filename)
                img = Image.open(img_path).convert('L')  # 转换为灰度图
                img = img.resize((28, 28))
                img_array = np.array(img)
                images.append(img_array)
                labels.append(0)  # 0表示印刷文字
    
    # 加载手写文字图片
    handwritten_dir = os.path.join(data_dir, 'handwritten')
    if os.path.exists(handwritten_dir):
        for filename in os.listdir(handwritten_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(handwritten_dir, filename)
                img = Image.open(img_path).convert('L')  # 转换为灰度图
                img = img.resize((28, 28))
                img_array = np.array(img)
                images.append(img_array)
                labels.append(1)  # 1表示手写文字
    
    return np.array(images), np.array(labels)

def train_model():
    """
    训练CNN模型
    """
    # 构建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("模型结构:")
    print(model)
    
    # 这里可以加载实际数据
    # 由于没有实际数据集，我们创建一些示例数据用于演示
    print("创建示例数据进行演示...")
    # 创建示例数据 (1000个样本)
    X = np.random.random((1000, 28, 28)) * 255
    y = np.random.randint(0, 2, (1000, 1))
    
    # 数据归一化
    X = X.astype('float32') / 255.0
    
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # 添加通道维度
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 训练模型
    print("开始训练模型...")
    num_epochs = 10
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # 评估模型
    print(f'验证集准确率: {val_accuracies[-1]:.4f}')
    
    # 保存模型
    '''
    torch.save(model.state_dict(), 'text_classification_model.pth')
    print("模型已保存为 'text_classification_model.pth'")
    '''
    # 封装训练历史
    history = {
        'loss': train_losses,
        'accuracy': train_accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    }
    
    return model, history

def plot_training_history(history): #要用的话把开头plt的注释给去了
    """
    绘制训练历史
    """
    # 绘制准确率
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制损失
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Test Loss')
    plt.title('Model Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    # 训练模型
    model, history = train_model()
    
    # 绘制训练历史，要用的话把开头plt的注释给去了
    plot_training_history(history)
    
    print("训练完成!")