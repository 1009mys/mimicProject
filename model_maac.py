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

from transformers import AutoModel, BertModel, BertConfig

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

        self.conv = []
        self.bn = []
        self.relu = []

        self.conv.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups))
        self.bn.append(nn.BatchNorm1d(out_channels))
        self.relu.append(nn.ReLU())

        for i in range(r-1):
            self.conv.append(nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, groups=groups))
            self.bn.append(nn.BatchNorm1d(out_channels))
            self.relu.append(nn.ReLU())

        self.se = SEBlock1d(out_channels, r)
        self.reluF = nn.ReLU()

        self.conv = nn.ModuleList(self.conv)
        self.bn = nn.ModuleList(self.bn)
        self.relu = nn.ModuleList(self.relu)

    def forward(self, x):

        for i in range(len(self.conv)):
            x = self.conv[i](x)
            x = self.bn[i](x)
            x = self.relu[i](x)

        x = x * self.se(x)
        x = self.reluF(x)

        return x

class LinearBLock(nn.Module):
    def __init__(self, in_channels, out_channels, r=4):
        super().__init__()

        self.fc = []
        self.relu = []

        self.fc.append(nn.Linear(in_channels, out_channels))
        self.relu.append(nn.ReLU())

        for i in range(r-1):
            self.fc.append(nn.Linear(out_channels, out_channels))
            self.relu.append(nn.ReLU())

        self.fc = nn.ModuleList(self.fc)
        self.relu = nn.ModuleList(self.relu)

        #self.se = SEBlock1d(out_channels, r)
        #self.reluF = nn.ReLU()

    def forward(self, x):

        for i in range(len(self.fc)):
            x = self.fc[i](x)
            x = self.relu[i](x)

        #x = x * self.se(x)
        #x = self.reluF(x)

        return x

class LSTMBlock(nn.Module):
    def __init__(self, channels, hidden_size, num_layers, dropout=0.5):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size = channels, 
            hidden_size = hidden_size, 
            num_layers = num_layers, 
            dropout=dropout, 
            batch_first=True)

    def forward(self, x):
        
        x, _ = self.lstm(x)

        return x

class encoder(nn.Module):
    def __init__(self, encoder_pretrained=True):
        super().__init__()

        if encoder_pretrained == True:
            self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            self.encoder = BertModel(BertConfig(
                vocab_size=28996,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                pad_token_id=0,
            ))

    def forward(self, 
        x_CC_token_input_ids, 
        x_CC_token_attention_mask, 
        x_CC_token_token_type_ids):
        
        x_CC_input = {
            'input_ids': x_CC_token_input_ids, 
            'token_type_ids': x_CC_token_attention_mask, 
            'attention_mask': x_CC_token_token_type_ids
            }

        x_CC = self.encoder(**x_CC_input)

        return x_CC.pooler_output

class MAAC(nn.Module):
    def __init__(self, encoderModel=None):
        super().__init__()

        self.CC = []
        self.numerical = []
        self.sbp = []
        self.dbp = []
        self.o2sat = []
        self.resparate = []
        self.heartrate = []

        self.total = []
        
        if encoderModel == None:
            raise Exception("encoderModel is None")
        
        self.CC.append(encoderModel)

        self.sbp.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=1,
            dropout=0.2))

        self.dbp.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=1,
            dropout=0.2))

        self.o2sat.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=1,
            dropout=0.2))

        self.resparate.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=1,
            dropout=0.2))

        self.heartrate.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=1,
            dropout=0.2))

        self.numerical.append(LinearBLock(in_channels=233, out_channels=10, r=2))

        self.total.append(LinearBLock(in_channels=828, out_channels=2, r=1))
        #self.total.append(LinearBLock(in_channels=4096, out_channels=4096, r=4))
        #self.total.append(LinearBLock(in_channels=4096, out_channels=2, r=1))

        self.CC =        nn.ModuleList(self.CC)
        self.numerical = nn.ModuleList(self.numerical)
        self.sbp =       nn.ModuleList(self.sbp)
        self.dbp =       nn.ModuleList(self.dbp)
        self.o2sat =     nn.ModuleList(self.o2sat)
        self.resparate = nn.ModuleList(self.resparate)
        self.heartrate = nn.ModuleList(self.heartrate)
        self.total =     nn.ModuleList(self.total)

    def forward(self, 
        x_CC_token_input_ids, 
        x_CC_token_attention_mask, 
        x_CC_token_token_type_ids, 
        x_dbp, 
        x_sbp, 
        x_o2sat, 
        x_resparate, 
        x_heartrate, 
        x_numerical):

        
        for i in range(len(self.CC)):
            x_CC = self.CC[i](x_CC_token_input_ids, x_CC_token_attention_mask, x_CC_token_token_type_ids)

        for i in range(len(self.dbp)):
            x_dbp = self.dbp[i](x_dbp)

        for i in range(len(self.sbp)):
            x_sbp = self.sbp[i](x_sbp)

        for i in range(len(self.o2sat)):
            x_o2sat = self.o2sat[i](x_o2sat)

        for i in range(len(self.resparate)):
            x_resparate = self.resparate[i](x_resparate)

        for i in range(len(self.heartrate)):
            x_heartrate = self.heartrate[i](x_heartrate)

        for i in range(len(self.numerical)):
            x_numerical = self.numerical[i](x_numerical)

        """
        print("=========================================")

        print(x_dbp.shape)
        print(x_sbp.shape)
        print(x_o2sat.shape)
        print(x_resparate.shape)
        print(x_heartrate.shape)
        print(x_numerical.shape)

        print("=========================================")

        x_sbp = x_sbp.view(x_sbp.size(0), -1)
        x_dbp = x_dbp.view(x_dbp.size(0), -1)
        x_o2sat = x_o2sat.view(x_o2sat.size(0), -1)
        x_resparate = x_resparate.view(x_resparate.size(0), -1)
        x_heartrate = x_heartrate.view(x_heartrate.size(0), -1)
        x_numerical = x_numerical.view(x_numerical.size(0), -1)

        
        print(x_dbp.shape)
        print(x_sbp.shape)
        print(x_o2sat.shape)
        print(x_resparate.shape)
        print(x_heartrate.shape)
        print(x_numerical.shape)

        print("=========================================")
        """

        x_CC = x_CC.view(x_CC.size(0), 1, x_CC.size(1))

        #print(torch.tensor(x_CC).shape)

        x_total = torch.cat((x_CC, x_sbp, x_dbp, x_o2sat, x_resparate, x_heartrate, x_numerical), dim=2)


        for i in range(len(self.total)):
            x_total = self.total[i](x_total)

        x_total = x_total.view(x_total.size(0), -1)

        return x_total
        



if __name__=="__main__":
    
    print(print(torchsummary.summary(MAAC().cuda(), input_size=[
        (16,30), 
        (16,30), 
        (16,30), 
        (16, 1, 10), 
        (16, 1, 10), 
        (16, 1, 10), 
        (16, 1, 10), 
        (16, 1, 10), 
        (16, 1, 223)], device="cuda")))