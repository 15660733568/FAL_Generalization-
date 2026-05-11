import torch
import torch.nn as nn
import torch.nn.functional as F

# 通道注意力模块（SE块）
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock1D, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channels, length)
        w = x.mean(-1)  # (batch, channels)
        w = self.fc1(w)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        w = w.unsqueeze(-1)  # (batch, channels, 1)
        return x * w

# 残差块
class ResidualSEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ResidualSEBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock1D(channels, reduction)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += identity
        out = self.relu(out)
        return out

# 主体网络
class CustomCNN1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CustomCNN1D, self).__init__()
        self.init_channels = 128  # 通道数可以调大
        self.preconv = nn.Conv1d(1, self.init_channels, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm1d(self.init_channels)
        self.relu = nn.ReLU()
        self.resblock1 = ResidualSEBlock1D(self.init_channels)
        self.resblock2 = ResidualSEBlock1D(self.init_channels)
        self.pool = nn.MaxPool1d(2)
        self._fc_input_size = self.get_fc_input_size(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(self._fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self._initialize_weights()

    def get_fc_input_size(self, input_dim):
        with torch.no_grad():
            x = torch.zeros(1, 1, input_dim)
            x = self.preconv(x)
            x = self.bn0(x)
            x = self.relu(x)
            x = self.resblock1(x)
            x = self.resblock2(x)
            x = self.pool(x)
            return x.view(1, -1).shape[1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (batch, input_dim)
        x = x.unsqueeze(1)  # 变成 (batch, 1, input_dim)
        x = self.preconv(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
