import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import messagebox
import os
import time
import cv2

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 定义 LeNet 模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. 训练模型
def train_model():
    # 数据预处理：训练集变换（包含数据增强）
    transform = transforms.Compose([
        transforms.RandomRotation(10), # 随机旋转 ±10 度
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)), # 随机缩放 0.9-1.1 倍
        transforms.ToTensor(), # 转换为张量
        transforms.Normalize((0.1307,), (0.3081,)) # ImageNet 标准化 归一化
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 下载数据集
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    # 加载数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    #实例化模型
    model = LeNet().to(device)
    #实例化损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(15):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/15], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    torch.save(model.state_dict(), 'mnist_lenet.pth')
    print("模型已保存为 mnist_lenet.pth")

# 3. 手写板界面
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别器")
        self.root.geometry("400x500")

        # 初始化模型
        self.model = LeNet().to(device)
        if os.path.exists('mnist_lenet.pth'):
            self.model.load_state_dict(torch.load('mnist_lenet.pth', map_location=device))
            self.model.eval()
        else:
            messagebox.showwarning("警告", "未找到模型！请先训练模型。")
            self.root.quit()

        # 创建画布（白色背景）
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack(pady=10)

        # 创建图像对象
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        # 笔触粗细滑动条
        self.pen_size = tk.Scale(self.root, from_=4, to=12, orient=tk.HORIZONTAL, label="笔粗细")
        self.pen_size.set(8)
        self.pen_size.pack()

        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last)

        # 调试标签
        self.debug_label = tk.Label(self.root, text="绘制数字", font=("Arial", 10))
        self.debug_label.pack()

        # 按钮
        frame = tk.Frame(self.root)
        frame.pack(pady=5)
        self.predict_btn = tk.Button(frame, text="预测", command=self.predict)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        self.clear_btn = tk.Button(frame, text="清除", command=self.clear)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn = tk.Button(frame, text="保存图像", command=self.save_image)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # 结果标签
        self.result_label = tk.Label(self.root, text="预测：无", font=("Arial", 14))
        self.result_label.pack(pady=10)

        self.last_x, self.last_y = None, None
        self.root.after(500, self.update_prediction)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            self.canvas.create_line((self.last_x, self.last_y, x, y), fill='black', width=self.pen_size.get(), capstyle='round')
            self.draw.line((self.last_x, self.last_y, x, y), fill=0, width=self.pen_size.get())
            self.debug_label.config(text=f"绘制位置：({x}, {y})")
        self.last_x, self.last_y = x, y

    def reset_last(self, event):
        self.last_x, self.last_y = None, None
        self.debug_label.config(text="绘制数字")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="预测：无")
        self.debug_label.config(text="绘制数字")

    def save_image(self):
        filename = f"digit_{int(time.time())}.png"
        self.image.save(filename)
        messagebox.showinfo("提示", f"图像已保存为 {filename}")

    def update_prediction(self):
        if self.image.getbbox():
            self.predict()
        self.root.after(500, self.update_prediction)

    def predict(self):
        img = self.image
        bbox = img.getbbox()
        if bbox is None:
            messagebox.showerror("错误", "未绘制数字！")
            return
        left, top, right, bottom = bbox
        margin = 20
        left = max(0, left - margin)
        top = max(0, top - margin)
        right = min(280, right + margin)
        bottom = min(280, bottom + margin)
        img = img.crop((left, top, right, bottom))

        size = max(img.width, img.height)
        new_img = Image.new("L", (size, size), 255)
        offset_x = (size - img.width) // 2
        offset_y = (size - img.height) // 2
        new_img.paste(img, (offset_x, offset_y))

        img = new_img.resize((28, 28), Image.LANCZOS)
        img_np = np.array(img, dtype=np.float32)
        _, img_np = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY)
        img_array = img_np / 255.0
        img_array = 1.0 - img_array  # 反转颜色
        img_array = (img_array - 0.1307) / 0.3081
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            prediction = output.argmax(dim=1).item()
            confidence = probs[prediction].item()
            self.result_label.config(text=f"预测：{prediction} (置信度：{confidence:.2%})")

# 4. 主程序
if __name__ == "__main__":
    if not os.path.exists('mnist_lenet.pth'):
        print("正在训练模型...")
        train_model()
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()