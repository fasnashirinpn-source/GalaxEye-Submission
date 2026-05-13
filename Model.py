import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SiameseUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def encoder(self, x):
        x1 = self.enc1(x)
        p1 = self.pool1(x1)
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        x3 = self.enc3(p2)
        return x1, x2, x3

    def forward(self, pre, post):
        pre1, pre2, pre3 = self.encoder(pre)
        post1, post2, post3 = self.encoder(post)

        diff3 = torch.abs(pre3 - post3)
        x = self.up1(diff3)

        diff2 = torch.abs(pre2 - post2)
        x = torch.cat([x, diff2], dim=1)
        x = self.dec1(x)

        x = self.up2(x)
        diff1 = torch.abs(pre1 - post1)
        x = torch.cat([x, diff1], dim=1)
        x = self.dec2(x)

        return torch.sigmoid(self.final(x))
