import torch
import torch.nn as nn


class conv_bn_re(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, is_re=True):
        super(conv_bn_re, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.re = nn.ReLU()
        self.is_re = is_re

    def forward(self, x):
        if self.is_re:
            return self.re(self.bn(self.conv(x)))
        else:
            return self.bn(self.conv(x))


class basicblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, padding=1):
        super(basicblock, self).__init__()
        self.conv1 = conv_bn_re(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = conv_bn_re(out_channels, out_channels, kernel_size, 1, padding, is_re=False)
        if stride == 2:
            self.downsample = conv_bn_re(in_channels, out_channels, kernel_size=1, stride=2, is_re=False)
        else:
            self.downsample = None
        self.re = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.downsample:
            x = self.downsample(x)
        return self.re(y + x)


class SElayer(nn.Module):
    def __init__(self,channel,reduction=16):
        super(SElayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.func = nn.Sequential(
            nn.Conv2d(
                in_channels=channel,
                out_channels=int(channel/reduction)+1,
                kernel_size=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=int(channel/reduction)+1,
                out_channels=channel,
                kernel_size=1,
            ),
            nn.Sigmoid()
        )

    def forward(self,x):
        atten_x = self.pool(x)
        atten_x = self.func(atten_x).view(x.shape[0],-1,1,1)
        y = x * atten_x
        return y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam(nn.Module):
    def __init__(self,channels):
        super(cbam, self).__init__()
        self.cattn = SElayer(channels)
        self.sattn = SpatialAttention()

    def forward(self,x):
        out = self.cattn(x)
        out = self.sattn(out) * out
        return out


class SCVfliter(nn.Module):
    def __init__(self, dict_len, embed_size, num_hiddens, num_layers, nclass):
        super(SCVfliter, self).__init__()
        self.emb = nn.Embedding(dict_len, embed_size)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            basicblock(16, 32, 2, 3, 1),
            nn.MaxPool2d(1, 2),
            basicblock(32, 64, 2, 3, 1),
            cbam(64),
            basicblock(64, 128, 2, 3, 1),
            nn.MaxPool2d(1, 2),
            basicblock(128, 256, 2, 3, 1),
            cbam(256),
            basicblock(256, 512, 2, 3, 1),
            nn.MaxPool2d(1, 2),
            basicblock(512, 512, 1, 3, 1),
            cbam(512),
        )
        self.lstm = nn.LSTM(
            input_size=118,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
         )

        self.fc1 = nn.Linear(2 * num_hiddens, num_hiddens)
        self.re = nn.ReLU()
        self.fc2 = nn.Linear(num_hiddens, nclass)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, return_layer=0):
        emb = self.emb(x)
        if return_layer == 1:
            return emb
        emb = self.conv(emb.unsqueeze(1))
        emb = emb.squeeze(dim=3)
        if return_layer == 2:
            return emb
        out, _ = self.lstm(emb)
        out = out.permute(1, 0, 2)
        encoding = out[-1]
        if return_layer == 3:
            return encoding
        fearures = self.re(self.drop(self.fc1(encoding)))
        outs = self.fc2(fearures)

        return outs


if __name__ == '__main__':
    x = torch.ones((2, 30100), dtype=int)
    net = SCVfliter(6, 20, 100, 2, 2)
    print(net(x).shape)
