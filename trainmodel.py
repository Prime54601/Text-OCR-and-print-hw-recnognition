import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image, ImageFilter
# 新增导入用于绘图
import matplotlib.pyplot as plt

from dataLoader import *

# -----------------------------------------------------------
# 1. 假设的数据加载函数 (请替换为你实际的模块导入或定义)
# -----------------------------------------------------------
# from your_module import load_hw_data, generate_images_numpy

# 为了演示代码可运行，这里模拟一下这两个函数 (实际使用时请删除这部分)
# current_dir = os.getcwd()

# def load_hw_data(data_dir_hw, trn_count_hw, val_count_hw):
#     print("正在加载手写数据...")
#     # 模拟数据: (64x64 随机噪声图片, 标签 1)
#     train = [(np.random.randint(0, 255, (64, 64), dtype=np.uint8), 1) for _ in range(trn_count_hw)]
#     val = [(np.random.randint(0, 255, (64, 64), dtype=np.uint8), 1) for _ in range(val_count_hw)]
#     return train, val

# def generate_images_numpy(trn_count, val_count):
#     print("正在生成印刷数据...")
#     # 模拟数据: (64x64 随机噪声图片, 标签 0)
#     train = [(np.random.randint(0, 255, (64, 64), dtype=np.uint8), 0) for _ in range(trn_count)]
#     val = [(np.random.randint(0, 255, (64, 64), dtype=np.uint8), 0) for _ in range(val_count)]
#     return train, val

# -----------------------------------------------------------
# 2. 自定义 Dataset 类
# -----------------------------------------------------------
class CharClassificationDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list: 列表，每个元素是 (numpy_image, label)
            transform: torchvision.transforms
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_np, label = self.data_list[idx]
        
        # 将 Numpy 转换为 PIL Image 以便使用 torchvision transforms
        # 假设输入是灰度图 (H, W) 或 (H, W, C)
        img = Image.fromarray(img_np)
        
        # 如果是 RGB，可能需要转灰度，这里默认转为 'L' (灰度)
        if img.mode != 'L':
            img = img.convert('L')

        # =============================================
        # 【新增修改】 2. 条件锐化：仅针对手写数据 (label == 1)
        # =============================================
        # 注意：这个步骤必须在 resize 之前进行，以保留原始细节
        if label == 1:
            # 使用 PIL 的内置锐化滤镜
            img = img.filter(ImageFilter.SHARPEN)
            # 如果觉得不够锐，可以取消下面这行的注释再锐化一次
            # img = img.filter(ImageFilter.SHARPEN) 

        if self.transform:
            img = self.transform(img)
            
        # 标签转换为 Tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label

# -----------------------------------------------------------
# 3. 构建 CNN 模型
# -----------------------------------------------------------
class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()
        
        # 卷积层提取特征
        self.features = nn.Sequential(
            # Conv1: 输入 1通道(灰度), 输出 16通道
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 128x128 -> 64x64
            
            # Conv2: 16 -> 32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64x64 -> 32x32
            
            # Conv3: 32 -> 64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            # 添加额外的卷积层来进一步减小尺寸
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )
        
        # 全连接层进行分类
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), # 根据最后特征图大小 8x8
            nn.ReLU(),
            nn.Dropout(0.5), # 防止过拟合
            nn.Linear(128, 2) # 输出 2 类 (0: 印刷, 1: 手写)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------------------------------------
# 4. 主训练流程
# -----------------------------------------------------------
def main():
    # 配置参数
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    IMG_SIZE = 128  # 统一图片大小
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # A. 准备数据
    # 调用你的函数 (请确保你的函数已经定义或导入)
    hw_train, hw_val = load_hw_data(
        data_dir_hw=f"{current_dir}/HWDB1.1tst_gnt", 
        trn_count_hw=10000, 
        val_count_hw=1000
    )
    
    pr_train, pr_val = generate_images_numpy(
        trn_count=10000, 
        val_count=1000
    )

    # 合并数据
    train_data_raw = hw_train + pr_train
    # print (train_data_raw[1001][1])
    # print(train_data_raw[10121][1])
    # print(train_data_raw[14021][1])
    # print(train_data_raw[6021][1])
    # exit()
    val_data_raw = hw_val + pr_val
    
    print(f"训练集总数: {len(train_data_raw)}, 验证集总数: {len(val_data_raw)}")

    # 定义预处理
    # 这一步非常重要：因为手写数据和生成数据的原始尺寸可能不同，必须统一 resize
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # 强制缩放到 128x128
        transforms.ToTensor(),                   # 转为 Tensor 并归一化到 [0, 1]
    ])

    # 实例化 Dataset 和 DataLoader
    train_dataset = CharClassificationDataset(train_data_raw, transform=transform)
    val_dataset = CharClassificationDataset(val_data_raw, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # B. 初始化模型
    model = BinaryCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 新增: 创建用于存储训练历史的列表
    train_losses = []
    val_accuracies = []

    # C. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # print(labels)
            # error
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        # D. 验证循环
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                print(predicted)
                print(labels)
                error
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 新增: 记录训练历史
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)

    # E. 保存模型
    save_path = "/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/models/handwritten_vs_printed_cnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")

    # 新增: 绘制训练历史
    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(12, 4))
    
    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # 绘制验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/models/training_history.png')
    plt.show()

if __name__ == '__main__':
    main()