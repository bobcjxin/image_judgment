# 完整的模型训练套路 by gpu'
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


class Train(nn.Module):
    def __init__(self):
        super(Train, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            # nn.ReLU(True),
            # nn.Dropout(0.2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            # nn.ReLU(True),
            # nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 加载数据集
train_datasets = datasets.CIFAR10(root='cifar10_data', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.RandomHorizontalFlip(),    # 水平（左右）翻转
                                      transforms.RandomVerticalFlip(),      # 垂直（上下）反转
                                      transforms.RandomRotation(15),        # 旋转
                                      transforms.CenterCrop(32)
                                  ]))
# print(len(train_datasets))

# 分割数据集
val_len, train_len = int(len(train_datasets)*0.2), int(len(train_datasets)*0.8)
val_datasets, train_datasets = random_split(train_datasets, [val_len, train_len])
# print(len(val_datasets), len(train_datasets))
print('训练集长度：%d' % len(train_datasets))
print('验证集长度：%d' % len(val_datasets))
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=64, shuffle=True)

# 测试集
test_datasets = datasets.CIFAR10(root='cifar10_data', train=False, download=True,
                                 transform=transforms.ToTensor())

print('测试集长度：%d' % len(test_datasets))

test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)

# 实例化网络模型
train = Train()
if torch.cuda.is_available():
    train.cuda()

# 损失函数
loss = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss.cuda()

# 优化器
optimizer = torch.optim.Adam(train.parameters(), lr=0.001)

# tensorboard
writer = SummaryWriter(log_dir='./logs')

# 开始训练
best_acc = 0.0
fitting = 0
start = time.time()
for epoch in range(100):
    running_loss = 0.0
    print('------开始第%d次训练-------' % epoch)
    train.train()
    for i, data in enumerate(train_loader):
        imgs, targets = data
        if torch.cuda.is_available():
            imgs, targets = imgs.cuda(), targets.cuda()
        outputs = train(imgs)
        los = loss(outputs, targets)
        optimizer.zero_grad()
        los.backward()
        optimizer.step()
        running_loss += los.item()
        if i % 100 == 0:
            print('%d, %.3f' % (i, running_loss))
    print('train loss: %.3f' % running_loss)
    writer.add_scalar('train_loss', running_loss, epoch)

    train.eval()

    with torch.no_grad():
        correct = 0
        running_loss = 0
        for data in val_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs, targets = imgs.cuda(), targets.cuda()
            outputs = train(imgs)
            running_loss += loss(outputs, targets).item()
            correct += (outputs.argmax(1) == targets).sum()
        print('val loss: %.3f' % running_loss)
        cur_acc = 100 * correct/len(test_datasets)
        print('Accuracy of the network on the val images: %.3f %%' % (100 * correct / len(test_datasets)))
        if cur_acc > best_acc:
            best_acc = cur_acc
            torch.save(train.state_dict(), './models/cifra.pth')
            fitting = 0
            print('best acc: %.3f, epoch:%d' % (best_acc, epoch))
        writer.add_scalar('val_loss', running_loss, epoch)
        writer.add_scalar('val_accuracy', 100 * correct/len(test_datasets), epoch)
    print('use time: %.2f sec' % (time.time() - start))
    if fitting > 10:
        break
    else:
        fitting += 1

train.load_state_dict(torch.load('./models/cifra.pth'))
with torch.no_grad():
    correct = 0
    running_loss = 0
    for data in test_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs, targets = imgs.cuda(), targets.cuda()
        outputs = train(imgs)
        running_loss += loss(outputs, targets).item()
        correct += (outputs.argmax(1) == targets).sum()
    print('test loss: %.3f' % running_loss)
    cur_acc = 100 * correct / len(test_datasets)
    print('Accuracy of the network on the 10000 test images: %.3f %%' % (100 * correct / len(test_datasets)))


writer.close()
