from turtle import forward
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets
import torchsummary

class SEBlock1d(nn.Module):
    def __init__(self, in_channels, r=4): 
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool1d((1,))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels * r),
            nn.ReLU(),
            nn.Linear(in_channels * r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1)
        return x

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, r=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu3 = nn.ReLU()

        self.se = SEBlock1d(out_channels, r)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = x * self.se(x)

        x = self.relu(x)

        return x


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()

        channels = [1, 3, 5, 7, 9]
        
        self.floor1 = CNNBlock(channels[0], channels[1], 3, 1, 1)
        self.floor2 = CNNBlock(channels[1], channels[2], 3, 1, 1)
        self.floor3 = CNNBlock(channels[2], channels[3], 3, 1, 1)
        self.floor4 = CNNBlock(channels[3], channels[4], 3, 1, 1)

        self.linear1 = nn.Linear(2142, 4096)
        self.BN1 = nn.BatchNorm1d(4096)
        self.dropout1 = nn.Dropout(0.2)
        self.ReLu1 = nn.ReLU()

        self.linear2 = nn.Linear(4096, 4096)
        self.BN2 = nn.BatchNorm1d(4096)
        self.dropout2 = nn.Dropout(0.2)
        self.ReLu2 = nn.ReLU()

        self.out = nn.Linear(4096, 2)
        self.outRelu = nn.ReLU()

    def forward(self, x):
        x = self.floor1(x)
        x = self.floor2(x)
        x = self.floor3(x)
        x = self.floor4(x)

        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.BN1(x)
        x = self.dropout1(x)
        x = self.ReLu1(x)

        x = self.linear2(x)
        x = self.BN2(x)
        x = self.dropout2(x)
        x = self.ReLu2(x)

        output = self.out(x)
        output = self.outRelu(output)
        return output

class TestModel2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Linear(26, 1024)
        self.BN1 = nn.BatchNorm1d(1)
        self.dropout1 = nn.Dropout(0.2)

        self.linear2 = nn.Linear(1024, 1024)
        self.BN2 = nn.BatchNorm1d(1)
        self.dropout2 = nn.Dropout(0.2)

        self.out = nn.Linear(1024, 2)
        self.outRelu = nn.ReLU()


    def forward(self, x):
        
        x = self.linear1(x)
        x = self.BN1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.BN2(x)
        x = self.dropout2(x)

        output = self.out(x)
        output = self.outRelu(output)


        output = output.view(output.size(0), -1)
        
        
        return output


if __name__=="__main__":
    print(print(torchsummary.summary(TestModel().cuda(), input_size=(1,26))))