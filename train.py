# 完整的模型训练套路 by gpu'
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

class ResBlk(nn.Module):
    expansion = 4  # 扩张4倍

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        """
        :param in_channels: (int) number of input channels
        :param out_channels: (int) number of output channels
        :param stride: (int) stride of the convolution
        :param downsample: (nn.Module) downsample
        :param groups: (int) groups of the convolution
        :param base_width: (int) base width of the convolution
        :param dilation:  (int) dilation factor of the convolution
        """
        super(ResBlk, self).__init__()
        norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64)) * groups  # 中间层通道数

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, groups=groups, padding=dilation, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, out_channels*self.expansion, kernel_size=1)
        self.bn3 = norm_layer(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x    # [B, in_channel, H, W]

        out = self.relu(self.bn1(self.conv1(x)))    # [B, width, H, W]
        out = self.relu(self.bn2(self.conv2(out)))  # [B, width, int((H-1)/stride+1), int((W-1)/stride+1)]
        out = self.bn3(self.conv3(out))             # [B, 4*out_channel, int((H-1)/stride+1), int((W-1)/stride+1)]

        if self.downsample is not None:
            identity = self.downsample(x)           # 统一size

        out += identity    # 值相加，不改变size
        out = self.relu(out)

        return out  # [B, in_channel, H, W]  ->  [B, 4*out_channel, int((H-1)/stride+1), int((W-1)/stride+1)]


class ResTrain(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResTrain, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.out_channels = 64  # 公共输出通道数
        self.dilation = 1   # 卷积核扩张参数

        self.groups = 1
        self.base_width = 64    # block基础通道数

        self.conv1 = nn.Conv2d(3, self.out_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(3, 64)
        self.layer2 = self.make_layer(8, 128, stride=2, dilate=False)
        self.layer3 = self.make_layer(36, 256, stride=2, dilate=False)
        self.layer4 = self.make_layer(3, 512, stride=2, dilate=False)
        self.avgpool = nn.AvgPool2d(1)
        self.fc1 = nn.Linear(512 * ResBlk.expansion, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, blocks, channels, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None   # < class nn.Module >
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.out_channels != channels * ResBlk.expansion:
            # 把identy的size设为与out一致
            downsample = nn.Sequential(
                nn.Conv2d(self.out_channels, channels * ResBlk.expansion, kernel_size=1, stride=stride),
                norm_layer(channels * ResBlk.expansion),
            )

        layers = []
        layers.append(
            ResBlk(
                self.out_channels, channels, stride, downsample, self.groups, self.base_width, previous_dilation
            )
        )
        # -> [B, 4*channels, H/stride, W/stride]
        self.out_channels = channels * ResBlk.expansion     # 4*channel, 更新out_channels用作下一个in_channel
        for _ in range(1, blocks):
            layers.append(
                ResBlk(
                    self.out_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation
                )
            )
        return nn.Sequential(*layers)   # -> [B, 4*channels, H/stride, W/stride]

    def forward(self, x):
        # resize_transform = transforms.Resize((224, 224))
        # x = resize_transform(x)
        # print(x.shape)

        x = self.conv1(x)       # [64, 3, 32, 32] -> [64, 64, 16, 16]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # -> [64, 64, 8, 8]

        x = self.layer1(x)      # [64, 256, 8, 8]
        x = self.layer2(x)      # [64, 512, 4, 4]
        x = self.layer3(x)      # [64, 1024, 2, 2]
        x = self.layer4(x)      # [64, 2048, 1, 1]
        x = self.avgpool(x)     # [64, 2048, 1, 1]

        x = torch.flatten(x, 1)     # [64, 2048*1*1]

        # FC layer
        x = self.fc1(x)     # [64, 256]
        x = self.relu(x)
        x = self.fc2(x)     # [64, 64]
        x = self.relu(x)
        x = self.fc3(x)     # [64, 10]

        return x


res_train = ResTrain(10)
# print(res_train)

class Train(nn.Module):
    def __init__(self):
        super(Train, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # [b, 32, 32, 32]
            nn.ReLU(True),
            nn.MaxPool2d(2),    # [b, 32, 16, 16]

            # nn.Dropout(0.2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),    # [b, 32, 16, 16]
            nn.ReLU(True),
            nn.MaxPool2d(2),    # [b, 32, 8, 8]

            # nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),    # [b, 64, 8, 8]
            nn.MaxPool2d(2),    # [b, 64, 4, 4]
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
# train = Train()
train = res_train
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
# train.load_state_dict(torch.load('./models/res152.pth'))
for epoch in range(200):
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

    # 验证
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
            torch.save(train.state_dict(), './models/res152.pth')
            fitting = 0
            print('best acc: %.3f, epoch:%d' % (best_acc, epoch))
        writer.add_scalar('val_loss', running_loss, epoch)
        writer.add_scalar('val_accuracy', 100 * correct/len(test_datasets), epoch)
    print('use time: %.2f sec' % (time.time() - start))
    if fitting > 30:
        break
    else:
        fitting += 1

# 测试
train.load_state_dict(torch.load('./models/res152.pth'))
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
