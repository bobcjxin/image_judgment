# Acc Log
```python
# v1.0
class Train(nn.Module):
    def __init__(self):
        super(Train, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# Accuracy of the network on the 10000 test images: 67.240 %
```

```python
# V1.5 - ADD RELU
class Train(nn.Module):
    def __init__(self):
        super(Train, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),  # [b, 32, 32, 32]
            nn.ReLU(True),
            nn.MaxPool2d(2),    # [b, 32, 16, 16]
            nn.Conv2d(32, 32, kernel_size=5, padding=2),    # [b, 32, 16, 16]
            nn.ReLU(True),
            nn.MaxPool2d(2),    # [b, 32, 8, 8]
            nn.Conv2d(32, 64, kernel_size=5, padding=2),    # [b, 64, 8, 8]
            nn.MaxPool2d(2),    # [b, 64, 4, 4]
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

# Accuracy of the network on the 10000 test images: 67.770 %
```

```python
# v2.0 - ResNet152
class ResBlk(nn.Module):
    expansion = 4  # 扩张4倍

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(ResBlk, self).__init__()
        norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64)) * groups

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
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResTrain(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResTrain, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.out_channels = 64
        self.dilation = 1   # 卷积核扩张参数

        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(3, self.out_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = norm_layer(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(3, 64)
        self.layer2 = self.make_layer(8, 128, stride=2, dilate=False)
        self.layer3 = self.make_layer(36, 256, stride=2, dilate=False)
        self.layer4 = self.make_layer(3, 512, stride=2, dilate=False)
        self.avgpool = nn.AvgPool2d(1)
        self.fc = nn.Linear(512 * ResBlk.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, blocks, channels, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None 
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.out_channels != channels * ResBlk.expansion:
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
        self.out_channels = channels * ResBlk.expansion
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
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)       # [b, 64, 16, 16]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [b, 64, 8, 8]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Accuracy of the network on the 10000 test images: 76.170 %
```

```python
# v2.5 - ResNet152: optimize FC layer
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
        # print(out.shape)
        out = self.relu(self.bn2(self.conv2(out)))  # [B, width, int((H-1)/stride+1), int((W-1)/stride+1)]
        # print(out.shape)
        out = self.bn3(self.conv3(out))             # [B, 4*out_channel, int((H-1)/stride+1), int((W-1)/stride+1)]

        # print(identity.shape)
        # print(out.shape)
        if self.downsample is not None:
            identity = self.downsample(x)           # 统一size

        print(identity.shape)
        print(out.shape)

        out += identity    # 值相加，不改变size
        out = self.relu(out)

        print(out.shape)

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
        """

        :param blocks: (int) number of layers
        :param channels:
        :param stride:
        :param dilate:
        :return:
        """
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

# Accuracy of the network on the 10000 test images: 78.820 %
```