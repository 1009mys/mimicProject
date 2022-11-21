from turtle import forward
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets
import torchsummary

from transformers import AutoModel, BertModel, BertConfig
import math

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
        self.mp = []
        self.relu = []

        self.conv.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups))
        self.bn.append(nn.BatchNorm1d(out_channels))
        self.mp.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.relu.append(nn.ReLU())

        for i in range(r-1):
            self.conv.append(nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, groups=groups))
            self.bn.append(nn.BatchNorm1d(out_channels))
            self.mp.append(nn.MaxPool1d(kernel_size=2, stride=2))
            self.relu.append(nn.ReLU())

        self.se = SEBlock1d(out_channels, r)
        self.reluF = nn.ReLU()

        self.conv = nn.ModuleList(self.conv)
        self.bn = nn.ModuleList(self.bn)
        self.mp = nn.ModuleList(self.mp)
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
    def __init__(self, in_channels, out_channels, r=4, output=False, bias=True):
        super().__init__()

        self.fc = []
        self.relu = []
        self.bn = []
        self.do = []

        self.fc.append(nn.Linear(in_channels, out_channels, bias=bias))
        self.bn.append(nn.BatchNorm1d(out_channels))
        self.do.append(nn.Dropout(0.2))
        #self.relu.append(nn.ReLU())

        
        if output == True:
            self.relu.append(nn.Sigmoid())
        else:
            self.relu.append(nn.ReLU())
        
        for i in range(r-1):
            self.fc.append(nn.Linear(out_channels, out_channels, bias=bias))
            self.bn.append(nn.BatchNorm1d(out_channels))
            self.do.append(nn.Dropout(0.2))
            #self.relu.append(nn.ReLU())
            
            if i == r-2 and output == True:
                self.relu.append(nn.Sigmoid())
            else:
                self.relu.append(nn.ReLU())
            

        self.fc = nn.ModuleList(self.fc)
        self.relu = nn.ModuleList(self.relu)
        self.bn = nn.ModuleList(self.bn)
        self.do = nn.ModuleList(self.do)

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
            #self.encoder = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
            self.encoder = BertModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        ff = self.pe[:x.size(0)]
        x = torch.cat([x, ff], dim=2)
        return self.dropout(x)

class MACCwithTransformer_onlyTriage(nn.Module):
    def __init__(self, CC_encoderModel=None):
        super().__init__()

        self.CC_encoder = CC_encoderModel
        
        """
        self.numerical_embedding = nn.Embedding(num_embeddings=233, embedding_dim=8, padding_idx=0)
        self.numerical_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=233*8, nhead=2), num_layers=2)

        self.Sequential_x_dbp_pe = PositionalEncoding(d_model=40, dropout=0.1, max_len=1024)
        self.Sequential_x_dbp_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=50, nhead=5), num_layers=2)

        self.Sequential_x_sbp_pe = PositionalEncoding(d_model=40, dropout=0.1, max_len=1024)
        self.Sequential_x_sbp_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=50, nhead=5), num_layers=2)
        
        self.Sequential_x_o2sat_pe = PositionalEncoding(d_model=40, dropout=0.1, max_len=1024)
        self.Sequential_x_o2sat_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=50, nhead=5), num_layers=2)

        self.Sequential_x_resparate_pe = PositionalEncoding(d_model=40, dropout=0.1, max_len=1024)
        self.Sequential_x_resparate_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=50, nhead=5), num_layers=2)

        self.Sequential_x_heartrate_pe = PositionalEncoding(d_model=40, dropout=0.1, max_len=1024)
        self.Sequential_x_heartrate_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=50, nhead=5), num_layers=2)
        """

        self.x_heartrate_embedding      = nn.Embedding(num_embeddings=10, embedding_dim=1, padding_idx=0)
        self.x_resparate_embedding      = nn.Embedding(num_embeddings=10, embedding_dim=1, padding_idx=0)
        self.x_o2sat_embedding          = nn.Embedding(num_embeddings=10, embedding_dim=1, padding_idx=0)
        self.x_sbp_embedding            = nn.Embedding(num_embeddings=10, embedding_dim=1, padding_idx=0)
        self.x_dbp_embedding            = nn.Embedding(num_embeddings=10, embedding_dim=1, padding_idx=0)

        self.Bicarbonate_embedding      = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Creatinine_embedding       = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Glucose_embedding          = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Hematocrit_embedding       = nn.Embedding(num_embeddings=8,  embedding_dim=1, padding_idx=0)
        self.Hemoglobin_embedding       = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Platelets_embedding        = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Potassium_embedding        = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Sodium_embedding           = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Urea_Nitrogen_embedding    = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.white_blood_cell_embedding = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.pCO2_embedding             = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.pH_embedding               = nn.Embedding(num_embeddings=12, embedding_dim=1, padding_idx=0)
        self.Bilirubin_embedding        = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)

        self.x_heartrate_encoder        = PositionalEncoding(d_model=10, dropout=0.1, max_len=1024)
        self.x_resparate_encoder        = PositionalEncoding(d_model=10, dropout=0.1, max_len=1024)
        self.x_o2sat_encoder            = PositionalEncoding(d_model=10, dropout=0.1, max_len=1024)
        self.x_sbp_encoder              = PositionalEncoding(d_model=10, dropout=0.1, max_len=1024)
        self.x_dbp_encoder              = PositionalEncoding(d_model=10, dropout=0.1, max_len=1024)

        self.Bicarbonate_encoder        = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Creatinine_encoder         = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Glucose_encoder            = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Hematocrit_encoder         = PositionalEncoding(d_model=8,  dropout=0.1, max_len=1024)
        self.Hemoglobin_encoder         = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Platelets_encoder          = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Potassium_encoder          = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Sodium_encoder             = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Urea_Nitrogen_encoder      = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.white_blood_cell_encoder   = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.pCO2_encoder               = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.pH_encoder                 = PositionalEncoding(d_model=12, dropout=0.1, max_len=1024)
        self.Bilirubin_encoder          = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)


        

        self.total_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=2882, nhead=11), num_layers=2)

        self.fc = LinearBLock(2882, 1, 1, output=True)

    def forward(self, 
        x_CC_token_input_ids, 
        x_CC_token_attention_mask, 
        x_CC_token_token_type_ids, 
        x_heartrate, 
        x_resparate, 
        x_o2sat, 
        x_sbp, 
        x_dbp, 
        x_numerical1, 
        Bicarbonate, 
        Creatinine, 
        Glucose, 
        Hematocrit, 
        Platelet, 
        Potassium, 
        Sodium, 
        Urea_Nitrogen, 
        white_blood_cell, 
        pCO2, 
        pH, 
        Bilirubin, 
        x_numerical2):

        x_CC = self.CC_encoder(x_CC_token_input_ids, x_CC_token_attention_mask, x_CC_token_token_type_ids)
        x_CC = x_CC.view(x_CC.size(0), 1, x_CC.size(1))

        x_total = torch.cat((
            x_CC,
            x_heartrate, 
            x_resparate, 
            x_o2sat, 
            x_sbp, 
            x_dbp, 
            x_numerical1, 
            Bicarbonate, 
            Creatinine, 
            Glucose, 
            Hematocrit, 
            Platelet, 
            Potassium, 
            Sodium, 
            Urea_Nitrogen, 
            white_blood_cell, 
            pCO2, 
            pH, 
            Bilirubin, 
            x_numerical2))

        x_total = self.total_encoder(x_total)

        x_total = x_total.view(x_total.size(0), 1, x_total.size(1))
        output = self.fc(torch.cat((x_CC, x_total), dim=2))

        """
        x_CC = self.CC_encoder(x_CC_token_input_ids, x_CC_token_attention_mask, x_CC_token_token_type_ids)
        x_CC = x_CC.view(x_CC.size(0), 1, x_CC.size(1))
        x_numerical = x_numerical.to(torch.int64)
        x_numerical = self.numerical_embedding(x_numerical)
        x_numerical = x_numerical.view(x_numerical.shape[0], 1, x_numerical.shape[2]*x_numerical.shape[3])
        x_numerical = self.numerical_encoder(x_numerical)
         
        x_dbp = x_dbp.unsqueeze(1)
        x_sbp = x_sbp.unsqueeze(1)
        x_o2sat = x_o2sat.unsqueeze(1)
        x_resparate = x_resparate.unsqueeze(1)
        x_heartrate = x_heartrate.unsqueeze(1)

        x_sequential = torch.cat([x_dbp, x_sbp, x_o2sat, x_resparate, x_heartrate], dim=1)


        x_dbp = self.Sequential_x_dbp_pe(x_dbp)
        x_dbp = self.Sequential_x_dbp_encoder(x_dbp)

        x_sbp = self.Sequential_x_sbp_pe(x_sbp)
        x_sbp = self.Sequential_x_sbp_encoder(x_sbp)

        x_o2sat = self.Sequential_x_o2sat_pe(x_o2sat)
        x_o2sat = self.Sequential_x_o2sat_encoder(x_o2sat)

        x_resparate = self.Sequential_x_resparate_pe(x_resparate)
        x_resparate = self.Sequential_x_resparate_encoder(x_resparate)

        x_heartrate = self.Sequential_x_heartrate_pe(x_heartrate)
        x_heartrate = self.Sequential_x_heartrate_encoder(x_heartrate)

        x_total = torch.cat([x_CC, x_numerical, x_dbp, x_sbp, x_o2sat, x_resparate, x_heartrate], dim=2)



        #x_total = torch.cat([x_CC, x_numerical, x_sequential], dim=1)
        x_total = self.total_encoder(x_total)
        x_total = x_total.view(x_total.size(0), x_total.size(2))
        output = self.fc(x_total)
        """
        return output

class MACCwithTransformer(nn.Module):
    def __init__(self, CC_encoderModel=None):
        super().__init__()

        self.CC_encoder = CC_encoderModel
        
        """
        self.numerical_embedding = nn.Embedding(num_embeddings=233, embedding_dim=8, padding_idx=0)
        self.numerical_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=233*8, nhead=2), num_layers=2)

        self.Sequential_x_dbp_pe = PositionalEncoding(d_model=40, dropout=0.1, max_len=1024)
        self.Sequential_x_dbp_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=50, nhead=5), num_layers=2)

        self.Sequential_x_sbp_pe = PositionalEncoding(d_model=40, dropout=0.1, max_len=1024)
        self.Sequential_x_sbp_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=50, nhead=5), num_layers=2)
        
        self.Sequential_x_o2sat_pe = PositionalEncoding(d_model=40, dropout=0.1, max_len=1024)
        self.Sequential_x_o2sat_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=50, nhead=5), num_layers=2)

        self.Sequential_x_resparate_pe = PositionalEncoding(d_model=40, dropout=0.1, max_len=1024)
        self.Sequential_x_resparate_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=50, nhead=5), num_layers=2)

        self.Sequential_x_heartrate_pe = PositionalEncoding(d_model=40, dropout=0.1, max_len=1024)
        self.Sequential_x_heartrate_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=50, nhead=5), num_layers=2)
        """

        self.x_heartrate_embedding      = nn.Embedding(num_embeddings=10, embedding_dim=1, padding_idx=0)
        self.x_resparate_embedding      = nn.Embedding(num_embeddings=10, embedding_dim=1, padding_idx=0)
        self.x_o2sat_embedding          = nn.Embedding(num_embeddings=10, embedding_dim=1, padding_idx=0)
        self.x_sbp_embedding            = nn.Embedding(num_embeddings=10, embedding_dim=1, padding_idx=0)
        self.x_dbp_embedding            = nn.Embedding(num_embeddings=10, embedding_dim=1, padding_idx=0)

        self.Bicarbonate_embedding      = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Creatinine_embedding       = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Glucose_embedding          = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Hematocrit_embedding       = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Hemoglobin_embedding       = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Platelets_embedding        = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Potassium_embedding        = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Sodium_embedding           = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Urea_Nitrogen_embedding    = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.white_blood_cell_embedding = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.pO2_embedding              = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.pCO2_embedding             = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.pH_embedding               = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)
        self.Bilirubin_embedding        = nn.Embedding(num_embeddings=6,  embedding_dim=1, padding_idx=0)



        self.x_heartrate_encoder        = PositionalEncoding(d_model=10, dropout=0.1, max_len=1024)
        self.x_resparate_encoder        = PositionalEncoding(d_model=10, dropout=0.1, max_len=1024)
        self.x_o2sat_encoder            = PositionalEncoding(d_model=10, dropout=0.1, max_len=1024)
        self.x_sbp_encoder              = PositionalEncoding(d_model=10, dropout=0.1, max_len=1024)
        self.x_dbp_encoder              = PositionalEncoding(d_model=10, dropout=0.1, max_len=1024)

        self.Bicarbonate_encoder        = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Creatinine_encoder         = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Glucose_encoder            = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Hematocrit_encoder         = PositionalEncoding(d_model=8,  dropout=0.1, max_len=1024)
        self.Hemoglobin_encoder         = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Platelets_encoder          = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Potassium_encoder          = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Sodium_encoder             = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Urea_Nitrogen_encoder      = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.white_blood_cell_encoder   = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.pO2_encoder                = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.pCO2_encoder               = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.pH_encoder                 = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)
        self.Bilirubin_encoder          = PositionalEncoding(d_model=6,  dropout=0.1, max_len=1024)


        

        self.total_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=1176, nhead=3), num_layers=2)

        self.fc = LinearBLock(1944, 1, 1, output=True)

    def forward(self, 
        x_CC_token_input_ids, 
        x_CC_token_attention_mask, 
        x_CC_token_token_type_ids, 
        x_heartrate, 
        x_resparate, 
        x_o2sat, 
        x_sbp, 
        x_dbp, 
        x_numerical1, 
        Bicarbonate, 
        Creatinine, 
        Glucose, 
        Hematocrit, 
        Hemoglobin,
        Platelet, 
        Potassium, 
        Sodium, 
        Urea_Nitrogen, 
        white_blood_cell, 
        pO2,
        pCO2, 
        pH, 
        Bilirubin, 
        x_numerical2):

        x_CC = self.CC_encoder(x_CC_token_input_ids, x_CC_token_attention_mask, x_CC_token_token_type_ids)
        x_CC = x_CC.view(x_CC.size(0), 1, x_CC.size(1))


        x_heartrate = self.x_heartrate_encoder(x_heartrate)
        x_resparate = self.x_resparate_encoder(x_resparate)
        x_o2sat     = self.x_o2sat_encoder(x_o2sat)
        x_sbp       = self.x_sbp_encoder(x_sbp)
        x_dbp       = self.x_dbp_encoder(x_dbp)

        Bicarbonate     = self.Bicarbonate_encoder(Bicarbonate)
        Creatinine      = self.Creatinine_encoder(Creatinine)
        Glucose         = self.Glucose_encoder(Glucose)
        Hematocrit      = self.Hematocrit_encoder(Hematocrit)
        Hemoglobin      = self.Hemoglobin_encoder(Hemoglobin)
        Platelet        = self.Platelets_encoder(Platelet)
        Potassium       = self.Potassium_encoder(Potassium)
        Sodium          = self.Sodium_encoder(Sodium)
        Urea_Nitrogen   = self.Urea_Nitrogen_encoder(Urea_Nitrogen)
        white_blood_cell= self.white_blood_cell_encoder(white_blood_cell)
        pO2             = self.pO2_encoder(pO2)
        pCO2            = self.pCO2_encoder(pCO2)
        pH              = self.pH_encoder(pH)
        Bilirubin       = self.Bilirubin_encoder(Bilirubin)
        """
        print("after encoder")
        print(x_CC.shape)
        print(x_heartrate.shape)
        print(x_resparate.shape)
        print(x_o2sat.shape)
        print(x_sbp.shape)
        print(x_dbp.shape)
        print(x_numerical1.shape)
        print(Bicarbonate.shape)
        print(Creatinine.shape)
        print(Glucose.shape)
        print(Hematocrit.shape)
        print(Platelet.shape)
        print(Potassium.shape)
        print(Sodium.shape)
        print(Urea_Nitrogen.shape)
        print(white_blood_cell.shape)
        print(pCO2.shape)
        print(pH.shape)
        print(Bilirubin.shape)
        print(x_numerical2.shape)
        """


        x_total = torch.cat((
            x_CC,
            x_heartrate, 
            x_resparate, 
            x_o2sat, 
            x_sbp, 
            x_dbp, 
            x_numerical1, 
            Bicarbonate, 
            Creatinine, 
            Glucose, 
            Hematocrit, 
            Hemoglobin,
            Platelet, 
            Potassium, 
            Sodium, 
            Urea_Nitrogen, 
            white_blood_cell, 
            pO2,
            pCO2, 
            pH, 
            Bilirubin, 
            x_numerical2), dim=2)

        x_total = self.total_encoder(x_total)

        #print(x_total.shape)

        #x_total = x_total.view(x_total.size(0), 1, x_total.size(1))
        output = self.fc(torch.cat((x_CC, x_total), dim=2))

        """
        x_CC = self.CC_encoder(x_CC_token_input_ids, x_CC_token_attention_mask, x_CC_token_token_type_ids)
        x_CC = x_CC.view(x_CC.size(0), 1, x_CC.size(1))
        x_numerical = x_numerical.to(torch.int64)
        x_numerical = self.numerical_embedding(x_numerical)
        x_numerical = x_numerical.view(x_numerical.shape[0], 1, x_numerical.shape[2]*x_numerical.shape[3])
        x_numerical = self.numerical_encoder(x_numerical)
         
        x_dbp = x_dbp.unsqueeze(1)
        x_sbp = x_sbp.unsqueeze(1)
        x_o2sat = x_o2sat.unsqueeze(1)
        x_resparate = x_resparate.unsqueeze(1)
        x_heartrate = x_heartrate.unsqueeze(1)

        x_sequential = torch.cat([x_dbp, x_sbp, x_o2sat, x_resparate, x_heartrate], dim=1)


        x_dbp = self.Sequential_x_dbp_pe(x_dbp)
        x_dbp = self.Sequential_x_dbp_encoder(x_dbp)

        x_sbp = self.Sequential_x_sbp_pe(x_sbp)
        x_sbp = self.Sequential_x_sbp_encoder(x_sbp)

        x_o2sat = self.Sequential_x_o2sat_pe(x_o2sat)
        x_o2sat = self.Sequential_x_o2sat_encoder(x_o2sat)

        x_resparate = self.Sequential_x_resparate_pe(x_resparate)
        x_resparate = self.Sequential_x_resparate_encoder(x_resparate)

        x_heartrate = self.Sequential_x_heartrate_pe(x_heartrate)
        x_heartrate = self.Sequential_x_heartrate_encoder(x_heartrate)

        x_total = torch.cat([x_CC, x_numerical, x_dbp, x_sbp, x_o2sat, x_resparate, x_heartrate], dim=2)



        #x_total = torch.cat([x_CC, x_numerical, x_sequential], dim=1)
        x_total = self.total_encoder(x_total)
        x_total = x_total.view(x_total.size(0), x_total.size(2))
        output = self.fc(x_total)
        """
        return output

class MAAC_onlyTriage(nn.Module):
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
        
        """
        if encoderModel == None:
            raise Exception("encoderModel is None")
        """
        
        self.CC.append(encoderModel)

        self.sbp.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.dbp.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.o2sat.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))
        self.resparate.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.heartrate.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

            
        self.numerical.append(LinearBLock(
            in_channels=16,
            out_channels=512,
            r=4))



        
        

        

        
        self.total.append(LinearBLock(in_channels=1330, out_channels=4096, r=4, bias=True))
        #self.total.append(nn.Linear(in_features=2048, out_features=1))
        #self.total.append(nn.Sigmoid())
        self.total.append(LinearBLock(in_channels=4096, out_channels=1, r=1, output=True, bias=True))
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
        x_heartrate, 
        x_resparate, 
        x_o2sat, 
        x_sbp, 
        x_dbp, 
        x_gender,
        x_acuity,
        x_sequential):

        x_numerical = torch.cat((x_gender, x_acuity, x_sequential), dim=2)

        
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

        #x_CC = x_CC.view(x_CC.size(0), 1, x_CC.size(1))

        x_sbp = x_sbp.view(x_sbp.size(0), -1)
        x_dbp = x_dbp.view(x_dbp.size(0), -1)
        x_o2sat = x_o2sat.view(x_o2sat.size(0), -1)
        x_resparate = x_resparate.view(x_resparate.size(0), -1)
        x_heartrate = x_heartrate.view(x_heartrate.size(0), -1)



        x_numerical = x_numerical.view(x_numerical.size(0), -1)
        """
        print(x_CC.shape)
        print(x_dbp.shape)
        print(x_sbp.shape)
        print(x_o2sat.shape)
        print(x_resparate.shape)
        print(x_heartrate.shape)
        print(x_numerical.shape)
        print(Bicarbonate.shape)
        print(Creatinine.shape)
        print(Glucose.shape)
        print(Hematocrit.shape)
        print(Platelet.shape)
        print(Potassium.shape)
        print(Sodium.shape)
        print(Urea_Nitrogen.shape)
        print(white_blood_cell.shape)
        print(pCO2.shape)
        print(pH.shape)
        print(Bilirubin.shape)
        """

        
        x_total = torch.cat((x_CC, x_sbp, x_dbp, x_o2sat, x_resparate, x_heartrate, x_numerical), 1)

        
        for i in range(len(self.total)):
            x_total = self.total[i](x_total)
        
        #x_total = self.output(x_total)

        #x_total = self.total[0](x_total)
        #x_total_ = self.total[1](x_total)
        #x_total = self.total[2](x_total_)


        #x_total = x_total.view(x_total.size(0), -1)

        

        return x_total

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

        self.Bicarbonate = []
        self.Creatinine = []
        self.Glucose = []
        self.Hematocrit = []
        self.Hemoglobin = []
        self.Platelets = []
        self.Potassium = []
        self.Sodium = []
        self.UreaNitrogen = []
        self.white_blood_cell = []
        self.pO2 = []
        self.pCO2 = []
        self.pH = []
        self.Bilirubin = []
        self.x_numerical2 = []


        self.total = []
        
        """
        if encoderModel == None:
            raise Exception("encoderModel is None")
        """
        
        self.CC.append(encoderModel)

        self.sbp.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.dbp.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.o2sat.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))
        self.resparate.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.heartrate.append(LSTMBlock(
            channels = 10,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.Bicarbonate.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.Creatinine.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.Glucose.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.Hematocrit.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.Hemoglobin.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.Platelets.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.Potassium.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.Sodium.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.UreaNitrogen.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.white_blood_cell.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.pO2.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.pCO2.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.pH.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))

        self.Bilirubin.append(LSTMBlock(
            channels = 6,
            hidden_size=10,
            num_layers=4,
            dropout=0.2))
            


        self.numerical.append(LinearBLock(
            in_channels=138, 
            out_channels=512, 
            r=4))
        """
        self.numerical.append(CNNBlock(
            in_channels = 1,
            out_channels = 4,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            groups = 1,
            r = 4))
        self.numerical.append(CNNBlock(
            in_channels = 4,
            out_channels = 8,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            groups = 1,
            r = 4))
        """
        
        

        

        
        self.total.append(LinearBLock(in_channels=1470, out_channels=4096, r=4, bias=True))
        #self.total.append(nn.Linear(in_features=2048, out_features=1))
        #self.total.append(nn.Sigmoid())
        self.total.append(LinearBLock(in_channels=4096, out_channels=1, r=1, output=True, bias=True))
        #self.total.append(LinearBLock(in_channels=4096, out_channels=4096, r=4))
        #self.total.append(LinearBLock(in_channels=4096, out_channels=2, r=1))

        self.output = nn.Threshold(0.5, 0)

        self.CC =        nn.ModuleList(self.CC)
        self.numerical = nn.ModuleList(self.numerical)
        self.sbp =       nn.ModuleList(self.sbp)
        self.dbp =       nn.ModuleList(self.dbp)
        self.o2sat =     nn.ModuleList(self.o2sat)
        self.resparate = nn.ModuleList(self.resparate)
        self.heartrate = nn.ModuleList(self.heartrate)
        self.Bicarbonate = nn.ModuleList(self.Bicarbonate)
        self.Creatinine = nn.ModuleList(self.Creatinine)
        self.Glucose = nn.ModuleList(self.Glucose)
        self.Hematocrit = nn.ModuleList(self.Hematocrit)
        self.Hemoglobin = nn.ModuleList(self.Hemoglobin)
        self.Platelets = nn.ModuleList(self.Platelets)
        self.Potassium = nn.ModuleList(self.Potassium)
        self.Sodium = nn.ModuleList(self.Sodium)
        self.UreaNitrogen = nn.ModuleList(self.UreaNitrogen)
        self.white_blood_cell = nn.ModuleList(self.white_blood_cell)
        self.pO2 = nn.ModuleList(self.pO2)
        self.pCO2 = nn.ModuleList(self.pCO2)
        self.pH = nn.ModuleList(self.pH)
        self.Bilirubin = nn.ModuleList(self.Bilirubin)

        self.total =     nn.ModuleList(self.total)

    def forward(self, 
        x_CC_token_input_ids, 
        x_CC_token_attention_mask, 
        x_CC_token_token_type_ids, 
        x_heartrate, 
        x_resparate, 
        x_o2sat, 
        x_sbp, 
        x_dbp, 
        x_numerical1, 
        Bicarbonate, 
        Creatinine, 
        Glucose, 
        Hematocrit, 
        Hemoglobin,
        Platelet, 
        Potassium, 
        Sodium, 
        Urea_Nitrogen, 
        white_blood_cell, 
        pO2,
        pCO2, 
        pH, 
        Bilirubin, 
        x_numerical2):

        x_numerical = torch.cat((x_numerical1, x_numerical2), dim=2)

        
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
        
        for i in range(len(self.Bicarbonate)):
            Bicarbonate = self.Bicarbonate[i](Bicarbonate)

        for i in range(len(self.Creatinine)):
            Creatinine = self.Creatinine[i](Creatinine)

        for i in range(len(self.Glucose)):
            Glucose = self.Glucose[i](Glucose)

        for i in range(len(self.Hematocrit)):
            Hematocrit = self.Hematocrit[i](Hematocrit)
        
        for i in range(len(self.Hemoglobin)):
            Hemoglobin = self.Hemoglobin[i](Hemoglobin)

        for i in range(len(self.Platelets)):
            Platelet = self.Platelets[i](Platelet)

        for i in range(len(self.Potassium)):
            Potassium = self.Potassium[i](Potassium)

        for i in range(len(self.Sodium)):
            Sodium = self.Sodium[i](Sodium)

        for i in range(len(self.UreaNitrogen)):
            Urea_Nitrogen = self.UreaNitrogen[i](Urea_Nitrogen)

        for i in range(len(self.white_blood_cell)):
            white_blood_cell = self.white_blood_cell[i](white_blood_cell)
        
        for i in range(len(self.pO2)):
            pO2 = self.pO2[i](pO2)

        for i in range(len(self.pCO2)):
            pCO2 = self.pCO2[i](pCO2)

        for i in range(len(self.pH)):
            pH = self.pH[i](pH)

        for i in range(len(self.Bilirubin)):
            Bilirubin = self.Bilirubin[i](Bilirubin)



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

        #x_CC = x_CC.view(x_CC.size(0), 1, x_CC.size(1))

        x_sbp = x_sbp.view(x_sbp.size(0), -1)
        x_dbp = x_dbp.view(x_dbp.size(0), -1)
        x_o2sat = x_o2sat.view(x_o2sat.size(0), -1)
        x_resparate = x_resparate.view(x_resparate.size(0), -1)
        x_heartrate = x_heartrate.view(x_heartrate.size(0), -1)

        Bicarbonate = Bicarbonate.view(Bicarbonate.size(0), -1)
        Creatinine = Creatinine.view(Creatinine.size(0), -1)
        Glucose = Glucose.view(Glucose.size(0), -1)
        Hematocrit = Hematocrit.view(Hematocrit.size(0), -1)
        Hemoglobin = Hemoglobin.view(Hemoglobin.size(0), -1)
        Platelet = Platelet.view(Platelet.size(0), -1)
        Potassium = Potassium.view(Potassium.size(0), -1)
        Sodium = Sodium.view(Sodium.size(0), -1)
        Urea_Nitrogen = Urea_Nitrogen.view(Urea_Nitrogen.size(0), -1)
        white_blood_cell = white_blood_cell.view(white_blood_cell.size(0), -1)
        pO2 = pO2.view(pO2.size(0), -1)
        pCO2 = pCO2.view(pCO2.size(0), -1)
        pH = pH.view(pH.size(0), -1)
        Bilirubin = Bilirubin.view(Bilirubin.size(0), -1)


        x_numerical = x_numerical.view(x_numerical.size(0), -1)
        """
        print(x_CC.shape)
        print(x_dbp.shape)
        print(x_sbp.shape)
        print(x_o2sat.shape)
        print(x_resparate.shape)
        print(x_heartrate.shape)
        print(x_numerical.shape)
        print(Bicarbonate.shape)
        print(Creatinine.shape)
        print(Glucose.shape)
        print(Hematocrit.shape)
        print(Hemoglobin.shape)
        print(Platelet.shape)
        print(Potassium.shape)
        print(Sodium.shape)
        print(Urea_Nitrogen.shape)
        print(white_blood_cell.shape)
        print(pO2.shape)
        print(pCO2.shape)
        print(pH.shape)
        print(Bilirubin.shape)
        """

        
        x_total = torch.cat((x_CC, x_sbp, x_dbp, x_o2sat, x_resparate, x_heartrate, x_numerical, Bicarbonate, Creatinine, Glucose, Hematocrit, Hemoglobin, Platelet, Potassium, Sodium, Urea_Nitrogen, white_blood_cell, pO2, pCO2, pH, Bilirubin), 1)

        
        for i in range(len(self.total)):
            x_total = self.total[i](x_total)
        
        #x_total = self.output(x_total)

        #x_total = self.total[0](x_total)
        #x_total_ = self.total[1](x_total)
        #x_total = self.total[2](x_total_)


        #x_total = x_total.view(x_total.size(0), -1)

        

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