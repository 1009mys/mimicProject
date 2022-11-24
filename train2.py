from torch.utils.data import Dataset

from email.policy import default
import sys, getopt
from optparse import OptionParser
from xmlrpc.client import boolean

from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.nn as nn # layer들을 호출하기 위해서
import torch.optim as optim # optimization method를 사용하기 위해서
import torch.nn.init as init # weight initialization 해주기 위해서
import torchvision.datasets as dset # toy data들을 이용하기 위해서
import torchvision.transforms as transforms # pytorch 모델을 위한 데이터 변환을 위해
from torch.utils.data import DataLoader # train,test 데이터를 loader객체로 만들어주기 위해서

from model import TestModel, TestModel2
from model_maac import MAAC, encoder, MACCwithTransformer, MAAC_onlyTriage, MACCwithTransformer_onlyTriage
from dataloader import MimicLoader, MimicLoader_dataset1, MimicLoader_dataset1_onlyTriage
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

from copy import deepcopy

import datetime
from EarlyStopping import EarlyStopping
import torchsummary
from transformers import AutoTokenizer

import matplotlib.pyplot as plt

import pandas as pd


class MimicLoader_dataset1_onlyTriage_t(Dataset):
    def __init__(self,
        data_CC = False,
        data_Seq = False,
        annotations_file_x = None,
        annotations_file_y = None,
        transform=None, 
        train=True,
        test_size=0.2):

        if annotations_file_x == None or annotations_file_y == None:
            raise ValueError("annotations_file is None")

        self.data_CC = data_CC
        self.data_Seq = data_Seq

        mimic_label_x = pd.read_csv(annotations_file_x)#.head(1000)

        mimic_label_y = pd.read_csv(annotations_file_y)

        print(mimic_label_x)
        print(mimic_label_y)

        self.mimic_labels = mimic_label_x

        self.mimic_labels_y = mimic_label_y

        self.transform = transform

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


    def __len__(self):
        return len(self.mimic_labels)

    def CCtokenizer(self, x):
        
        x = self.tokenizer(x, padding='max_length', max_length=30)

        return x

    def __getitem__(self, idx):
        x_CC = self.mimic_labels.iloc[idx, 2]
        x_heartrate = self.mimic_labels.iloc[idx, 3:13]
        x_resparate = self.mimic_labels.iloc[idx, 13:23]
        x_o2sat = self.mimic_labels.iloc[idx, 23:33]
        x_sbp = self.mimic_labels.iloc[idx, 33:43]
        x_dbp = self.mimic_labels.iloc[idx, 43:53]
        #y = self.mimic_labels.iloc[idx, 53]
        y = self.mimic_labels_y.iloc[idx, 1]

        x_gender = self.mimic_labels.iloc[idx, 53:55]
        x_acuity = self.mimic_labels.iloc[idx, 60:65]
        x_sequential = self.mimic_labels.iloc[idx, 245:254]


        

        #x_CC = np.array(list(x_CC))
        x_heartrate = np.array(list(x_heartrate))
        x_resparate = np.array(list(x_resparate))
        x_o2sat = np.array(list(x_o2sat))
        x_sbp = np.array(list(x_sbp))
        x_dbp = np.array(list(x_dbp))
        x_gender = np.array(list(x_gender))
        x_acuity = np.array(list(x_acuity))
        x_sequential = np.array(list(x_sequential))
        
        y = np.array(y)

        x_CC_token = self.CCtokenizer(x_CC)

        x_dbp[x_dbp!=x_dbp] = 0.0
        x_sbp[x_sbp!=x_sbp] = 0.0
        x_o2sat[x_o2sat!=x_o2sat] = 0.0
        x_resparate[x_resparate!=x_resparate] = 0.0
        x_heartrate[x_heartrate!=x_heartrate] = 0.0

        x_heartrate.astype(np.float32)
        x_resparate.astype(np.float32)
        x_o2sat.astype(np.float32)
        x_sbp.astype(np.float32)
        x_dbp.astype(np.float32)
        x_gender.astype(np.float32)
        x_acuity.astype(np.float32)
        x_sequential.astype(np.float32)


        

        x_CC_token_input_ids = torch.Tensor(x_CC_token['input_ids'])
        x_CC_token_attention_mask = torch.Tensor(x_CC_token['attention_mask'])
        x_CC_token_token_type_ids = torch.Tensor(x_CC_token['token_type_ids'])

        x_dbp = torch.from_numpy(x_dbp)
        x_sbp = torch.from_numpy(x_sbp)
        x_o2sat = torch.from_numpy(x_o2sat)
        x_resparate = torch.from_numpy(x_resparate)
        x_heartrate = torch.from_numpy(x_heartrate)
        x_gender = torch.from_numpy(x_gender)
        x_acuity = torch.from_numpy(x_acuity)
        x_sequential = torch.from_numpy(x_sequential)
        


        y = torch.from_numpy(y)
            
        x_heartrate = torch.unsqueeze(x_heartrate,0)
        x_resparate = torch.unsqueeze(x_resparate,0)
        x_o2sat = torch.unsqueeze(x_o2sat,0)
        x_sbp = torch.unsqueeze(x_sbp,0)
        x_dbp = torch.unsqueeze(x_dbp,0)
        x_gender = torch.unsqueeze(x_gender, 0)
        x_acuity = torch.unsqueeze(x_acuity, 0)
        x_sequential = torch.unsqueeze(x_sequential, 0)
        
        

        return (x_CC_token_input_ids, 
            x_CC_token_attention_mask, 
            x_CC_token_token_type_ids, 
            x_heartrate, 
            x_resparate, 
            x_o2sat, 
            x_sbp, 
            x_dbp, 
            x_gender,
            x_acuity,
            x_sequential,
             y)

class MimicLoader_dataset1_t(Dataset):
    def __init__(self,
        data_CC = False,
        data_Seq = False,
        annotations_file_x = None,
        annotations_file_y = None,
        transform=None, 
        train=True,
        random_seed=42,
        test_size=0.2):

        if annotations_file_x == None or annotations_file_y == None:
            raise ValueError("annotations_file is None")

        self.data_CC = data_CC
        self.data_Seq = data_Seq

        mimic_label_x = pd.read_csv(annotations_file_x)#.head(1000)

        mimic_label_y = pd.read_csv(annotations_file_y)

        #print(mimic_label)

        print(mimic_label_x)
        print(mimic_label_y)

        self.mimic_labels = mimic_label_x

        self.mimic_labels_y = mimic_label_y

        self.transform = transform

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        #print(len(mimic_label))
        #print("")

    def __len__(self):
        return len(self.mimic_labels)

    def CCtokenizer(self, x):
        
        x = self.tokenizer(x, padding='max_length', max_length=30)

        return x

    def __getitem__(self, idx):
        x_CC = self.mimic_labels.iloc[idx, 2]
        x_heartrate = self.mimic_labels.iloc[idx, 3:13]
        x_resparate = self.mimic_labels.iloc[idx, 13:23]
        x_o2sat = self.mimic_labels.iloc[idx, 23:33]
        x_sbp = self.mimic_labels.iloc[idx, 33:43]
        x_dbp = self.mimic_labels.iloc[idx, 43:53]
        
        #y = self.mimic_labels.iloc[idx, 53]
        y = self.mimic_labels_y.iloc[idx, 1]

        x_numerical1 = self.mimic_labels.iloc[idx, 53:161]
        Bicarbonate = self.mimic_labels.iloc[idx, 161:167]
        Creatinine = self.mimic_labels.iloc[idx, 167:173]
        Glucose = self.mimic_labels.iloc[idx, 173:179]
        Hematocrit = self.mimic_labels.iloc[idx, 179:185]
        Hemoglobin = self.mimic_labels.iloc[idx, 185:191]
        Platelet = self.mimic_labels.iloc[idx, 191:197]
        Potassium = self.mimic_labels.iloc[idx, 197:203]
        Sodium = self.mimic_labels.iloc[idx, 203:209]
        Urea_Nitrogen = self.mimic_labels.iloc[idx, 209:215]
        white_blood_cell = self.mimic_labels.iloc[idx, 215:221]
        pO2 = self.mimic_labels.iloc[idx, 221:227]
        pCO2 = self.mimic_labels.iloc[idx, 227:233]
        pH = self.mimic_labels.iloc[idx, 233:239]
        Bilirubin = self.mimic_labels.iloc[idx, 239:245]
        x_numerical2 = self.mimic_labels.iloc[idx, 245:]

        #x_CC = np.array(list(x_CC))
        x_heartrate = np.array(list(x_heartrate))
        x_resparate = np.array(list(x_resparate))
        x_o2sat = np.array(list(x_o2sat))
        x_sbp = np.array(list(x_sbp))
        x_dbp = np.array(list(x_dbp))
        x_numerical1 = np.array(list(x_numerical1))
        Bicarbonate = np.array(list(Bicarbonate))
        Creatinine = np.array(list(Creatinine))
        Glucose = np.array(list(Glucose))
        Hematocrit = np.array(list(Hematocrit))
        Hemoglobin = np.array(list(Hemoglobin))
        Platelet = np.array(list(Platelet))
        Potassium = np.array(list(Potassium))
        Sodium = np.array(list(Sodium))
        Urea_Nitrogen = np.array(list(Urea_Nitrogen))
        white_blood_cell = np.array(list(white_blood_cell))
        pO2 = np.array(list(pO2))
        pCO2 = np.array(list(pCO2))
        pH = np.array(list(pH))
        Bilirubin = np.array(list(Bilirubin))
        x_numerical2 = np.array(list(x_numerical2))

        y = np.array(y)

        x_CC_token = self.CCtokenizer(x_CC)

        x_dbp[x_dbp!=x_dbp] = 0.0
        x_sbp[x_sbp!=x_sbp] = 0.0
        x_o2sat[x_o2sat!=x_o2sat] = 0.0
        x_resparate[x_resparate!=x_resparate] = 0.0
        x_heartrate[x_heartrate!=x_heartrate] = 0.0

        x_heartrate.astype(np.float32)
        x_resparate.astype(np.float32)
        x_o2sat.astype(np.float32)
        x_sbp.astype(np.float32)
        x_dbp.astype(np.float32)
        x_numerical1.astype(np.float32)
        Bicarbonate.astype(np.float32)
        Creatinine.astype(np.float32)
        Glucose.astype(np.float32)
        Hematocrit.astype(np.float32)
        Hemoglobin.astype(np.float32)
        Platelet.astype(np.float32)
        Potassium.astype(np.float32)
        Sodium.astype(np.float32)
        Urea_Nitrogen.astype(np.float32)
        white_blood_cell.astype(np.float32)
        pO2.astype(np.float32)
        pCO2.astype(np.float32)
        pH.astype(np.float32)
        Bilirubin.astype(np.float32)
        x_numerical2.astype(np.float32)

        x_CC_token_input_ids = torch.Tensor(x_CC_token['input_ids'])
        x_CC_token_attention_mask = torch.Tensor(x_CC_token['attention_mask'])
        x_CC_token_token_type_ids = torch.Tensor(x_CC_token['token_type_ids'])

        x_dbp = torch.from_numpy(x_dbp)
        x_sbp = torch.from_numpy(x_sbp)
        x_o2sat = torch.from_numpy(x_o2sat)
        x_resparate = torch.from_numpy(x_resparate)
        x_heartrate = torch.from_numpy(x_heartrate)
        x_numerical1 = torch.from_numpy(x_numerical1)
        Bicarbonate = torch.from_numpy(Bicarbonate)
        Creatinine = torch.from_numpy(Creatinine)
        Glucose = torch.from_numpy(Glucose)
        Hematocrit = torch.from_numpy(Hematocrit)
        Hemoglobin = torch.from_numpy(Hemoglobin)
        Platelet = torch.from_numpy(Platelet)
        Potassium = torch.from_numpy(Potassium)
        Sodium = torch.from_numpy(Sodium)
        Urea_Nitrogen = torch.from_numpy(Urea_Nitrogen)
        white_blood_cell = torch.from_numpy(white_blood_cell)
        pCO2 = torch.from_numpy(pCO2)
        pO2 = torch.from_numpy(pO2)
        pH = torch.from_numpy(pH)
        Bilirubin = torch.from_numpy(Bilirubin)
        x_numerical2 = torch.from_numpy(x_numerical2)


        y = torch.from_numpy(y)
            
        x_heartrate = torch.unsqueeze(x_heartrate,0)
        x_resparate = torch.unsqueeze(x_resparate,0)
        x_o2sat = torch.unsqueeze(x_o2sat,0)
        x_sbp = torch.unsqueeze(x_sbp,0)
        x_dbp = torch.unsqueeze(x_dbp,0)
        x_numerical1 = torch.unsqueeze(x_numerical1,0)
        Bicarbonate = torch.unsqueeze(Bicarbonate,0)
        Creatinine = torch.unsqueeze(Creatinine,0)
        Glucose = torch.unsqueeze(Glucose,0)
        Hematocrit = torch.unsqueeze(Hematocrit,0)
        Hemoglobin = torch.unsqueeze(Hemoglobin,0)
        Platelet = torch.unsqueeze(Platelet,0)
        Potassium = torch.unsqueeze(Potassium,0)
        Sodium = torch.unsqueeze(Sodium,0)
        Urea_Nitrogen = torch.unsqueeze(Urea_Nitrogen,0)
        white_blood_cell = torch.unsqueeze(white_blood_cell,0)
        pO2 = torch.unsqueeze(pO2,0)
        pCO2 = torch.unsqueeze(pCO2,0)
        pH = torch.unsqueeze(pH,0)
        Bilirubin = torch.unsqueeze(Bilirubin,0)
        x_numerical2 = torch.unsqueeze(x_numerical2,0)

        return (x_CC_token_input_ids, 
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
            x_numerical2, 
            y)

def trainEffNet(parser):

    (options, args) = parser.parse_args()

    batch_size = options.batch_size
    learning_rate = options.learning_rate
    num_epoch = options.num_epoch
    model = options.model
    #class_num = options.class_num
    data = options.data
    result_name = options.result_name
    loss_function = options.loss_function
    save_dir = options.save_dir
    """
    parser.add_option("--train_data_x", "-d", default=None, dest="train_data_x", type=str)
    parser.add_option("--train_data_y", "-d", default=None, dest="train_data_y", type=str)

    parser.add_option("--test_data_a_x", "-d", default=None, dest="test_a_x", type=str)
    parser.add_option("--test_data_b_x", "-d", default=None, dest="test_b_x", type=str)

    parser.add_option("--test_data_a_y", "-d", default=None, dest="test_a_y", type=str)
    parser.add_option("--test_data_b_y", "-d", default=None, dest="test_b_y", type=str)
    """

    train_data_x = options.train_data_x
    train_data_y = options.train_data_y

    test_a = options.test_a
    test_b = options.test_b


    if options.only_triage == 'True':
        only_triage = True
    else:
        only_triage = False


    if options.data_CC == 'True':
        data_CC = True
    else:
        data_CC = False

    if options.data_Seq == 'True':
        data_Seq = True
    else:
        data_Seq = False


    #pre_trained = options.pre_trained

    now = datetime.datetime.now()
    
    print("===========================================")
    print("Train Start")
    print("===========================================")
    # define the image transformation for trains_ds
    # in paper, using FiveCrop, normalize, horizontal reflection
    train_transformer = transforms.Compose([
                    #transforms.RandomHorizontalFlip(),
                    #transforms.Grayscale(1),
                    #transforms.RandomAffine(degrees=(0, 360)),
                    #transforms.GaussianBlur(kernel_size=3),
                    #transforms.RandomRotation(degrees=[0,360]),
                    #transforms.ColorJitter(brightness=0.2)
    ])

    if only_triage == False:
        mimicTrain = MimicLoader_dataset1_t(
            data_CC=data_CC, 
            data_Seq=data_Seq, 
            annotations_file_x=train_data_x, 
            annotations_file_y=train_data_y,
            transform=train_transformer, 
            train=True)
            
        mimicTest_a = MimicLoader_dataset1(
            data_CC=data_CC, 
            data_Seq=data_Seq,
            annotations_file=test_a, 
            train=False)
        
        mimicTest_b = MimicLoader_dataset1(
            data_CC=data_CC, 
            data_Seq=data_Seq,
            annotations_file=test_b, 
            train=False)
    else:
        mimicTrain = MimicLoader_dataset1_onlyTriage_t(
            data_CC=data_CC, 
            data_Seq=data_Seq, 
            annotations_file_x=train_data_x, 
            annotations_file_y=train_data_y,
            transform=train_transformer, 
            train=True)
        mimicTest_a = MimicLoader_dataset1_onlyTriage(
            data_CC=data_CC, 
            data_Seq=data_Seq,
            annotations_file=test_a,
            train=False)
        mimicTest_b = MimicLoader_dataset1_onlyTriage(
            data_CC=data_CC, 
            data_Seq=data_Seq,
            annotations_file=test_b, 
            train=False)
    
    
    #print(mimicTrain.__len__())


    # Data loader 객체 생성
    # 데이터 batch로 나눠주고 shuffle해주는 데이터 loader 객체 생성
    
    train_loader = DataLoader(mimicTrain,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=options.workers,
                              drop_last=True,
                              pin_memory=True
                              )
    test_loader_a = DataLoader(mimicTest_a,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=options.workers,
                             drop_last=False,
                             pin_memory=True
                             )
    test_loader_b = DataLoader(mimicTest_b,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=options.workers,
                                drop_last=False,
                                pin_memory=True
                                )
                             
    device = torch.device("cuda")
   
    if model == 'macc':
        if only_triage == False:
            encoderModel = encoder(encoder_pretrained=True)
            #print(encoderModel)
            model = MAAC(encoderModel)
        else:
            encoderModel = encoder(encoder_pretrained=True)
            #print(encoderModel)
            model = MAAC_onlyTriage(encoderModel)

    elif model == 'maccwithTransformer':
        if only_triage == False:
            encoderModel = encoder(encoder_pretrained=True)
            model = MACCwithTransformer(encoderModel)
        else:
            encoderModel = encoder(encoder_pretrained=True)
            model = MACCwithTransformer_onlyTriage(encoderModel)
        

    #print(model)

    NGPU = torch.cuda.device_count()
    

    model = nn.DataParallel(model)   # 4개의 GPU를 이용할 경우 pre_trained

    print("-------------------------")
    for i in range(NGPU):
        print(torch.cuda.get_device_name(i))
    print(sys.version)
    print("-------------------------")

    model = model.to(device)

    print(model)


    loss_func = None
    if loss_function == 'criterion':
        #loss_func = nn.CrossEntropyLoss()  # 크로스엔트로피 loss 객체, softmax를 포함함
        loss_func = nn.BCELoss()
    elif loss_function == 'MSE':
        loss_func = nn.MSELoss()
    else:
        raise Exception("올바른 loss함수가 아님!")

    

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    loss_list = []
    acc_list = []
    best_acc = 0
    best_f1 = 0
    best_acc_model = None 
    best_f1_model = None
   
    early_stopping = EarlyStopping(patience = 20, verbose = True)

    for epoch in range(num_epoch):
        model.train()
        if only_triage == False:
            for i , (x_CC_token_input_ids, 
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
                x_numerical2, 
                y) in enumerate(train_loader):

                x_CC_token_input_ids = x_CC_token_input_ids.to(device).long()
                x_CC_token_attention_mask = x_CC_token_attention_mask.to(device).long()
                x_CC_token_token_type_ids = x_CC_token_token_type_ids.to(device).long()
                x_heartrate = x_heartrate.to(device).float()
                x_resparate = x_resparate.to(device).float()
                x_o2sat = x_o2sat.to(device).float()
                x_sbp = x_sbp.to(device).float()
                x_dbp = x_dbp.to(device).float()
                x_numerical1 = x_numerical1.to(device).float()
                Bicarbonate = Bicarbonate.to(device).float()
                Creatinine = Creatinine.to(device).float()
                Glucose = Glucose.to(device).float()
                Hematocrit = Hematocrit.to(device).float()
                Hemoglobin = Hemoglobin.to(device).float()
                Platelet = Platelet.to(device).float()
                Potassium = Potassium.to(device).float()
                Sodium = Sodium.to(device).float()
                Urea_Nitrogen = Urea_Nitrogen.to(device).float()
                white_blood_cell = white_blood_cell.to(device).float()
                pO2 = pO2.to(device).float()
                pCO2 = pCO2.to(device).float()
                pH = pH.to(device).float()
                Bilirubin = Bilirubin.to(device).float()
                x_numerical2 = x_numerical2.to(device).float()


                y = y.to(device).float()

                output = model.forward(x_CC_token_input_ids, 
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
                    x_numerical2)

                output = output.view(output.size(0))

                loss = loss_func(output, y)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

        if only_triage == True:
            for i , (x_CC_token_input_ids, 
                x_CC_token_attention_mask, 
                x_CC_token_token_type_ids, 
                x_heartrate, 
                x_resparate, 
                x_o2sat, 
                x_sbp, 
                x_dbp, 
                x_gender,
                x_acuity, 
                x_sequential,
                y) in enumerate(train_loader):

                x_CC_token_input_ids = x_CC_token_input_ids.to(device).long()
                x_CC_token_attention_mask = x_CC_token_attention_mask.to(device).long()
                x_CC_token_token_type_ids = x_CC_token_token_type_ids.to(device).long()
                x_heartrate = x_heartrate.to(device).float()
                x_resparate = x_resparate.to(device).float()
                x_o2sat = x_o2sat.to(device).float()
                x_sbp = x_sbp.to(device).float()
                x_dbp = x_dbp.to(device).float()
                x_gender = x_gender.to(device).float()
                x_acuity = x_acuity.to(device).float()
                x_sequential = x_sequential.to(device).float()


                y = y.to(device).float()

                output = model.forward(x_CC_token_input_ids, 
                    x_CC_token_attention_mask, 
                    x_CC_token_token_type_ids, 
                    x_heartrate, 
                    x_resparate, 
                    x_o2sat, 
                    x_sbp, 
                    x_dbp, 
                    x_gender,
                    x_acuity,
                    x_sequential)

                output = output.view(output.size(0))

                loss = loss_func(output, y)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()


        print('epoch : ', epoch)
        #print('loss : ', loss.data)
        

        model.eval()
        test_loss = 0
        correct = 0
        

        guesses = np.array([])
        labels = np.array([])

        # a
        if only_triage == False:
            with torch.no_grad():
                for idx, (x_CC_token_input_ids, 
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
                    x_numerical2,
                    label) in enumerate(test_loader_a):

                    x_CC_token_input_ids = x_CC_token_input_ids.to(device).long()
                    x_CC_token_attention_mask = x_CC_token_attention_mask.to(device).long()
                    x_CC_token_token_type_ids = x_CC_token_token_type_ids.to(device).long()
                    x_heartrate = x_heartrate.to(device).float()
                    x_resparate = x_resparate.to(device).float()
                    x_o2sat = x_o2sat.to(device).float()
                    x_sbp = x_sbp.to(device).float()
                    x_dbp = x_dbp.to(device).float()
                    x_numerical1 = x_numerical1.to(device).float()
                    Bicarbonate = Bicarbonate.to(device).float()
                    Creatinine = Creatinine.to(device).float()
                    Glucose = Glucose.to(device).float()
                    Hematocrit = Hematocrit.to(device).float()
                    Hemoglobin = Hemoglobin.to(device).float()
                    Platelet = Platelet.to(device).float()
                    Potassium = Potassium.to(device).float()
                    Sodium = Sodium.to(device).float()
                    Urea_Nitrogen = Urea_Nitrogen.to(device).float()
                    white_blood_cell = white_blood_cell.to(device).float()
                    pO2 = pO2.to(device).float()
                    pCO2 = pCO2.to(device).float()
                    pH = pH.to(device).float()
                    Bilirubin = Bilirubin.to(device).float()
                    x_numerical2 = x_numerical2.to(device).float()

                    target = label.to(device).float()
                    
                    #print(x.shape)

                    # train데이터 셋 feedforwd 과정
                    output = model.forward(x_CC_token_input_ids, 
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
                        x_numerical2)

                    output = output.view(output.size(0))
                    lossT = loss_func(output, target)
                    test_loss +=  lossT.item()
                    #pred = output.argmax(dim=1, keepdim=True)
                    #pred = output.threshold(0.5, 1)
                    #s = score.argmax(dim=1, keepdim=True)

                    tmp1 = np.array(output.to('cpu'))
                    tmp2 = np.array(target.to('cpu'))
                    #tmp3 = np.array(s.to('cpu'))

                    tt1 = np.array(tmp1[:])
                    tt2 = np.array(tmp2[:])
                    #tt3 = np.array(tmp3[:])

                    guesses = np.append(guesses, tt1)
                    labels = np.append(labels, tt2)
                    #scores = np.append(scores, tt3)
        if only_triage == True:
            with torch.no_grad():
                for idx, (x_CC_token_input_ids, 
                    x_CC_token_attention_mask, 
                    x_CC_token_token_type_ids, 
                    x_heartrate, 
                    x_resparate, 
                    x_o2sat, 
                    x_sbp, 
                    x_dbp, 
                    x_gender,
                    x_acuity, 
                    x_sequential,
                    label) in enumerate(test_loader_a):

                    x_CC_token_input_ids = x_CC_token_input_ids.to(device).long()
                    x_CC_token_attention_mask = x_CC_token_attention_mask.to(device).long()
                    x_CC_token_token_type_ids = x_CC_token_token_type_ids.to(device).long()
                    x_heartrate = x_heartrate.to(device).float()
                    x_resparate = x_resparate.to(device).float()
                    x_o2sat = x_o2sat.to(device).float()
                    x_sbp = x_sbp.to(device).float()
                    x_dbp = x_dbp.to(device).float()
                    x_gender = x_gender.to(device).float()
                    x_acuity = x_acuity.to(device).float()
                    x_sequential = x_sequential.to(device).float()

                    target = label.to(device).float()
                    
                    #print(x.shape)

                    # train데이터 셋 feedforwd 과정
                    output = model.forward(x_CC_token_input_ids, 
                        x_CC_token_attention_mask, 
                        x_CC_token_token_type_ids, 
                        x_heartrate, 
                        x_resparate, 
                        x_o2sat, 
                        x_sbp, 
                        x_dbp, 
                        x_gender,
                        x_acuity,
                        x_sequential)

                    output = output.view(output.size(0))
                    lossT = loss_func(output, target)
                    test_loss +=  lossT.item()
                    #pred = output.argmax(dim=1, keepdim=True)
                    #pred = output.threshold(0.5, 1)
                    #s = score.argmax(dim=1, keepdim=True)

                    #tmp1 = np.array(output.to('cpu'))
                    #tmp2 = np.array(target.to('cpu'))
                    tmp1 = output.to('cpu').detach().numpy()
                    tmp2 = target.to('cpu').detach().numpy()

                    #tmp3 = np.array(s.to('cpu'))

                    tt1 = np.array(tmp1[:])
                    tt2 = np.array(tmp2[:])
                    #tt3 = np.array(tmp3[:])

                    guesses = np.append(guesses, tt1)
                    labels = np.append(labels, tt2)
                    #scores = np.append(scores, tt3)
        # b
        if only_triage == False:
            with torch.no_grad():
                for idx, (x_CC_token_input_ids, 
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
                    x_numerical2,
                    label) in enumerate(test_loader_b):

                    x_CC_token_input_ids = x_CC_token_input_ids.to(device).long()
                    x_CC_token_attention_mask = x_CC_token_attention_mask.to(device).long()
                    x_CC_token_token_type_ids = x_CC_token_token_type_ids.to(device).long()
                    x_heartrate = x_heartrate.to(device).float()
                    x_resparate = x_resparate.to(device).float()
                    x_o2sat = x_o2sat.to(device).float()
                    x_sbp = x_sbp.to(device).float()
                    x_dbp = x_dbp.to(device).float()
                    x_numerical1 = x_numerical1.to(device).float()
                    Bicarbonate = Bicarbonate.to(device).float()
                    Creatinine = Creatinine.to(device).float()
                    Glucose = Glucose.to(device).float()
                    Hematocrit = Hematocrit.to(device).float()
                    Hemoglobin = Hemoglobin.to(device).float()
                    Platelet = Platelet.to(device).float()
                    Potassium = Potassium.to(device).float()
                    Sodium = Sodium.to(device).float()
                    Urea_Nitrogen = Urea_Nitrogen.to(device).float()
                    white_blood_cell = white_blood_cell.to(device).float()
                    pO2 = pO2.to(device).float()
                    pCO2 = pCO2.to(device).float()
                    pH = pH.to(device).float()
                    Bilirubin = Bilirubin.to(device).float()
                    x_numerical2 = x_numerical2.to(device).float()

                    target = label.to(device).float()
                    
                    #print(x.shape)

                    # train데이터 셋 feedforwd 과정
                    output = model.forward(x_CC_token_input_ids, 
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
                        x_numerical2)

                    output = output.view(output.size(0))
                    lossT = loss_func(output, target)
                    test_loss +=  lossT.item()
                    #pred = output.argmax(dim=1, keepdim=True)
                    #pred = output.threshold(0.5, 1)
                    #s = score.argmax(dim=1, keepdim=True)

                    tmp1 = np.array(output.to('cpu'))
                    tmp2 = np.array(target.to('cpu'))
                    #tmp3 = np.array(s.to('cpu'))

                    tt1 = np.array(tmp1[:])
                    tt2 = np.array(tmp2[:])
                    #tt3 = np.array(tmp3[:])

                    guesses = np.append(guesses, tt1)
                    labels = np.append(labels, tt2)
                    #scores = np.append(scores, tt3)
        if only_triage == True:
            with torch.no_grad():
                for idx, (x_CC_token_input_ids, 
                    x_CC_token_attention_mask, 
                    x_CC_token_token_type_ids, 
                    x_heartrate, 
                    x_resparate, 
                    x_o2sat, 
                    x_sbp, 
                    x_dbp, 
                    x_gender,
                    x_acuity, 
                    x_sequential,
                    label) in enumerate(test_loader_b):

                    x_CC_token_input_ids = x_CC_token_input_ids.to(device).long()
                    x_CC_token_attention_mask = x_CC_token_attention_mask.to(device).long()
                    x_CC_token_token_type_ids = x_CC_token_token_type_ids.to(device).long()
                    x_heartrate = x_heartrate.to(device).float()
                    x_resparate = x_resparate.to(device).float()
                    x_o2sat = x_o2sat.to(device).float()
                    x_sbp = x_sbp.to(device).float()
                    x_dbp = x_dbp.to(device).float()
                    x_gender = x_gender.to(device).float()
                    x_acuity = x_acuity.to(device).float()
                    x_sequential = x_sequential.to(device).float()

                    target = label.to(device).float()
                    
                    #print(x.shape)

                    # train데이터 셋 feedforwd 과정
                    output = model.forward(x_CC_token_input_ids, 
                        x_CC_token_attention_mask, 
                        x_CC_token_token_type_ids, 
                        x_heartrate, 
                        x_resparate, 
                        x_o2sat, 
                        x_sbp, 
                        x_dbp, 
                        x_gender,
                        x_acuity,
                        x_sequential)

                    output = output.view(output.size(0))
                    lossT = loss_func(output, target)
                    test_loss +=  lossT.item()
                    #pred = output.argmax(dim=1, keepdim=True)
                    #pred = output.threshold(0.5, 1)
                    #s = score.argmax(dim=1, keepdim=True)

                    #tmp1 = np.array(output.to('cpu'))
                    #tmp2 = np.array(target.to('cpu'))
                    tmp1 = output.to('cpu').detach().numpy()
                    tmp2 = target.to('cpu').detach().numpy()

                    #tmp3 = np.array(s.to('cpu'))

                    tt1 = np.array(tmp1[:])
                    tt2 = np.array(tmp2[:])
                    #tt3 = np.array(tmp3[:])

                    guesses = np.append(guesses, tt1)
                    labels = np.append(labels, tt2)
                    #scores = np.append(scores, tt3)
                
        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        #guesses = guesses.astype(int)
        #labels = labels.astype(int)
        #guesses = guesses.astype(int)

        guesses = list(guesses)
        total = 0
        for g in guesses:
            total += g

        
        total /= len(guesses)

        print(guesses[:100])
        print(labels[:100])

        ff = [int(l) for l in labels]

        #print(guesses)
        guesses_ = deepcopy(guesses)
        guesses = [0 if guesses[i] <= 0.5 else 1 for i in range(len(guesses))]
        labels =  [0 if labels[i] <= 0.5 else 1 for i in range(len(labels))]
        #print(total)


        #guesses = list(guesses)
        #labels = list(labels)

        #scores = list(scores)

        #print(scores,'\n',labels)

        acc = accuracy_score(labels, guesses)
        f_score = f1_score(labels, guesses, average='macro')
        

        print(classification_report(labels, guesses, labels=[0,1]))

        """
        with open(save_dir + '/' + result_name + "_" + str(now) + "_epoch" + str(epoch) + ".txt", "w") as text_file:
            print("epoch:", epoch, file=text_file)
            print("test loss:", test_loss, file=text_file)
            print(classification_report(labels, guesses, digits=3), file=text_file)
            print(model, file=text_file)
            print("average", total, file=text_file)
        
        torch.save(best_acc_model, save_dir + '/' + result_name + "_" + str(now) + "_epoch" + str(epoch) + '.pt')
        """

        #misc (acc 계산, etc) 

        if acc > best_acc:
            best_acc = acc
            best_acc_model = deepcopy(model.state_dict())

            fprs, tprs, thresholds = roc_curve(np.array(labels), np.array(guesses_))
    
            # 대각선
            plt.plot([0,1],[0,1],label='STR')

            # ROC
            plt.plot(fprs,tprs,label='ROC')


            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend()
            plt.grid()
            plt.savefig(save_dir + '/' + result_name + "_" + str(now) + "_best_acc.png")
            plt.cla()


            with open(save_dir + '/' + result_name + "_" + str(now) + "_best_acc.txt", "w") as text_file:
                print("epoch:", epoch, file=text_file)
                print('roc auc value {}'.format(roc_auc_score(y_true=np.array(labels), y_score=np.array(guesses_))), file=text_file)
                print("test loss:", test_loss, file=text_file)
                print(classification_report(labels, guesses, digits=3), file=text_file)
                print(model, file=text_file)
                print("average", total, file=text_file)
            
            torch.save(best_acc_model,  save_dir + '/' + result_name + "_" + str(now) + '_best_acc.pt')

        if f_score > best_f1:
            best_f1 = f_score
            best_f1_model = deepcopy(model.state_dict())

            fprs, tprs, thresholds = roc_curve(np.array(labels), np.array(guesses_))
    
            # 대각선
            plt.plot([0,1],[0,1],label='STR')

            # ROC
            plt.plot(fprs,tprs,label='ROC')


            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend()
            plt.grid()
            plt.savefig(save_dir + '/' + result_name + "_" + str(now) + "_best_f1.png")
            plt.cla()

            
            with open( save_dir + '/' + result_name + "_" + str(now) + "_best_f1.txt", "w") as text_file:
                print("epoch:", epoch, file=text_file)
                print('roc auc value {}'.format(roc_auc_score(np.array(labels),np.array(guesses_))), file=text_file)
                print("test loss:", test_loss, file=text_file)
                print(classification_report(labels, guesses, digits=3), file=text_file)
                print(model, file=text_file)

            torch.save(best_f1_model,  save_dir + '/' + result_name + "_" + str(now) + '_best_f1.pt')

        #print('accuracy:', round(accuracy_score(labels, guesses), ndigits=3))
        #print('recall score:', round(recall_score(labels, guesses, average='micro'), ndigits=3))
        #print('precision score:', round(precision_score(labels, guesses, average='micro'), ndigits=3))
        #print('f1 score:', round(f1_score(labels, guesses, average='micro'), ndigits=3))

        print('-----------------')
    
        

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--batch", "-b", default=16, dest="batch_size", type=int)
    parser.add_option("--learning_rate", "-l", default=0.0001, dest="learning_rate", type=float)
    parser.add_option("--epochs", "-e", default=500, dest="num_epoch", type=int)
    parser.add_option("--model", "-m", default='0', dest="model", type=str)
    parser.add_option("--workers", "-w", default=1, dest="workers", type=int)
    #parser.add_option("--class_num", "-c", default=11, dest="class_num", type=int)
    parser.add_option("--data", "-d", default=None, dest="data", type=str)

    parser.add_option("--train_data_x",  default=None, dest="train_data_x", type=str)
    parser.add_option("--train_data_y",  default=None, dest="train_data_y", type=str)
    parser.add_option("--test_data_a", default=None, dest="test_a", type=str)
    parser.add_option("--test_data_b", default=None, dest="test_b", type=str)


    parser.add_option("--result_name", "-n", default="", dest="result_name", type=str)
    parser.add_option("--loss", default="criterion", dest="loss_function", type=str)
    #parser.add_option("--pre_trained", default=None, dest="pre_trained", type=int)
    #parser.add_option("--weight", "-W", default="", dest="weight", type=str)
    parser.add_option("--data_CC", default="False", dest="data_CC", type=str)
    parser.add_option("--data_Seq", default="False", dest="data_Seq", type=str)
    parser.add_option("--only_triage", default="False", dest="only_triage", type=str)
    parser.add_option("--save_dir", default="./result", dest="save_dir", type=str)
    trainEffNet(parser)
