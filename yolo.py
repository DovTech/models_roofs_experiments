import torch
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(ConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)

class YOLO(nn.Module):

    def __init__(self, **kwargs) -> None:
        super(YOLO, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.convs = self.__create_conv()
        self.fcs = self.__create_fc(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def __create_conv(self) -> nn.Sequential:
        layers = []
        #first step
        layers.append(ConvBlock(3, 64, kernel_size=7, stride=2, padding=3))
        layers.append(self.MaxPool())
        #second step
        layers.append(ConvBlock(64, 192, kernel_size=3, stride=1, padding=1))
        layers.append(self.MaxPool())
        #third step
        layers.append(ConvBlock(192, 128, kernel_size=1))
        layers.append(ConvBlock(128, 256, kernel_size=3, padding=1))
        layers.append(ConvBlock(256, 256, kernel_size=1))
        layers.append(ConvBlock(256, 512, kernel_size=3, padding=1))
        layers.append(self.MaxPool())
        #fourth step
        for _ in range(4):
            layers.append(ConvBlock(512, 256, kernel_size=1))
            layers.append(ConvBlock(256, 512, kernel_size=3, padding=1))
        layers.append(ConvBlock(512, 512, kernel_size=1))
        layers.append(ConvBlock(512, 1024, kernel_size=3, padding=1))
        layers.append(self.MaxPool())
        #fifth step
        for _ in range(2):
            layers.append(ConvBlock(1024, 512, kernel_size=1))
            layers.append(ConvBlock(512, 1024, kernel_size=3, padding=1))
        layers.append(ConvBlock(1024, 1024, kernel_size=3, padding=1))
        layers.append(ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1))
        #sixth step
        for _ in range(2):
            layers.append(ConvBlock(1024, 1024, kernel_size=3, padding=1))
        
        return nn.Sequential(*layers)
    
    def __create_fc(self, splits: int, boxes: int, classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * splits**2, 496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, splits**2 * (classes + boxes * 5))
        )