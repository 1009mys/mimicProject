from sre_parse import Tokenizer

from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import numpy as np
import torch

from transformers import AutoTokenizer

class MimicLoader_dataset1_onlyTriage(Dataset):
    def __init__(self,
        data_CC = False,
        data_Seq = False,
        annotations_file = None,
        transform=None, 
        train=True,
        random_seed=42,
        test_size=0.2):

        if annotations_file == None:
            raise ValueError("annotations_file is None")

        self.data_CC = data_CC
        self.data_Seq = data_Seq

        mimic_label = pd.read_csv(annotations_file)#.head(1000)

        print(mimic_label)

        mimic_labels_train, mimic_labels_test = train_test_split(mimic_label, test_size=test_size, random_state=random_seed)

        if train==True:
            self.mimic_labels = mimic_labels_train
        else:
            self.mimic_labels = mimic_labels_test

        self.transform = transform

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        print(len(mimic_label))
        print("")

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
        y = self.mimic_labels.iloc[idx, 53]

        x_gender = self.mimic_labels.iloc[idx, 54:56]
        x_acuity = self.mimic_labels.iloc[idx, 61:66]


        

        #x_CC = np.array(list(x_CC))
        x_heartrate = np.array(list(x_heartrate))
        x_resparate = np.array(list(x_resparate))
        x_o2sat = np.array(list(x_o2sat))
        x_sbp = np.array(list(x_sbp))
        x_dbp = np.array(list(x_dbp))
        x_gender = np.array(list(x_gender))
        x_acuity = np.array(list(x_acuity))
        
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
        


        y = torch.from_numpy(y)
            
        x_heartrate = torch.unsqueeze(x_heartrate,0)
        x_resparate = torch.unsqueeze(x_resparate,0)
        x_o2sat = torch.unsqueeze(x_o2sat,0)
        x_sbp = torch.unsqueeze(x_sbp,0)
        x_dbp = torch.unsqueeze(x_dbp,0)
        x_gender = torch.unsqueeze(x_gender, 0)
        x_acuity = torch.unsqueeze(x_acuity, 0)
        
        

        return (x_CC_token_input_ids, 
            x_CC_token_attention_mask, 
            x_CC_token_token_type_ids, 
            x_heartrate, 
            x_resparate, 
            x_o2sat, 
            x_sbp, 
            x_dbp, 
            x_gender,
            x_acuity, y)

class MimicLoader_dataset1(Dataset):
    def __init__(self,
        data_CC = False,
        data_Seq = False,
        annotations_file = None,
        transform=None, 
        train=True,
        random_seed=42,
        test_size=0.2):

        if annotations_file == None:
            raise ValueError("annotations_file is None")

        self.data_CC = data_CC
        self.data_Seq = data_Seq

        mimic_label = pd.read_csv(annotations_file)#.head(1000)

        print(mimic_label)

        mimic_labels_train, mimic_labels_test = train_test_split(mimic_label, test_size=test_size, random_state=random_seed)

        if train==True:
            self.mimic_labels = mimic_labels_train
        else:
            self.mimic_labels = mimic_labels_test

        self.transform = transform

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        print(len(mimic_label))
        print("")

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
        y = self.mimic_labels.iloc[idx, 53]
        x_numerical1 = self.mimic_labels.iloc[idx, 54:163]
        Bicarbonate = self.mimic_labels.iloc[idx, 163:169]
        Creatinine = self.mimic_labels.iloc[idx, 169:175]
        Glucose = self.mimic_labels.iloc[idx, 175:181]
        Hematocrit = self.mimic_labels.iloc[idx, 181:189]
        Platelet = self.mimic_labels.iloc[idx, 189:195]
        Potassium = self.mimic_labels.iloc[idx, 195:201]
        Sodium = self.mimic_labels.iloc[idx, 201:207]
        Urea_Nitrogen = self.mimic_labels.iloc[idx, 207:213]
        white_blood_cell = self.mimic_labels.iloc[idx, 213:219]
        pCO2 = self.mimic_labels.iloc[idx, 219:231]
        pH = self.mimic_labels.iloc[idx, 231:237]
        Bilirubin = self.mimic_labels.iloc[idx, 237:243]
        x_numerical2 = self.mimic_labels.iloc[idx, 243:]

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
        Platelet = np.array(list(Platelet))
        Potassium = np.array(list(Potassium))
        Sodium = np.array(list(Sodium))
        Urea_Nitrogen = np.array(list(Urea_Nitrogen))
        white_blood_cell = np.array(list(white_blood_cell))
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
        Platelet.astype(np.float32)
        Potassium.astype(np.float32)
        Sodium.astype(np.float32)
        Urea_Nitrogen.astype(np.float32)
        white_blood_cell.astype(np.float32)
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
        Platelet = torch.from_numpy(Platelet)
        Potassium = torch.from_numpy(Potassium)
        Sodium = torch.from_numpy(Sodium)
        Urea_Nitrogen = torch.from_numpy(Urea_Nitrogen)
        white_blood_cell = torch.from_numpy(white_blood_cell)
        pCO2 = torch.from_numpy(pCO2)
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
        Platelet = torch.unsqueeze(Platelet,0)
        Potassium = torch.unsqueeze(Potassium,0)
        Sodium = torch.unsqueeze(Sodium,0)
        Urea_Nitrogen = torch.unsqueeze(Urea_Nitrogen,0)
        white_blood_cell = torch.unsqueeze(white_blood_cell,0)
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
            Platelet, 
            Potassium, 
            Sodium, 
            Urea_Nitrogen, 
            white_blood_cell, 
            pCO2, 
            pH, 
            Bilirubin, 
            x_numerical2, 
            y)
        
# dataloader
class MimicLoader(Dataset):

    def __init__(self,
        data_CC = False,
        data_Seq = False,
        annotations_file = None,
        transform=None, 
        train=True,
        random_seed=42,
        test_size=0.2):

        if annotations_file == None:
            raise ValueError("annotations_file is None")

        self.data_CC = data_CC
        self.data_Seq = data_Seq

        mimic_label = pd.read_csv(annotations_file)#.head(1000)

        #print(mimic_label)

        mimic_labels_train, mimic_labels_test = train_test_split(mimic_label, test_size=test_size, random_state=random_seed)

        if train==True:
            self.mimic_labels = mimic_labels_train
        else:
            self.mimic_labels = mimic_labels_test

        self.transform = transform

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def __len__(self):
        return len(self.mimic_labels)

    def CCtokenizer(self, x):
        
        x = self.tokenizer(x, padding='max_length', max_length=30)

        return x

    def __getitem__(self, idx):

        if self.data_CC == True and self.data_Seq == True:
            y = self.mimic_labels.iloc[idx, 0]
            x_dbp = self.mimic_labels.iloc[idx, -10:]
            x_sbp = self.mimic_labels.iloc[idx, -20:-10]
            x_o2sat = self.mimic_labels.iloc[idx, -30:-20]
            x_resparate = self.mimic_labels.iloc[idx, -40:-30]
            x_heartrate = self.mimic_labels.iloc[idx, -50:-40]
            x_numerical = self.mimic_labels.iloc[idx, 2:-50]
            x_CC = self.mimic_labels.iloc[idx, 1]

            


            
            x_dbp = np.array(list(x_dbp))
            x_sbp = np.array(list(x_sbp))
            x_o2sat = np.array(list(x_o2sat))
            x_resparate = np.array(list(x_resparate))
            x_heartrate = np.array(list(x_heartrate))
            x_numerical = np.array(list(x_numerical))
            #x_numerical = np.array(map(str, list(x_numerical)), dtype=np.unicode_).flatten()
            
            #x_CC = np.array([x_CC])

            y = np.array(y)

            x_CC_token = self.CCtokenizer(x_CC)


            

            
            
            x_dbp[x_dbp!=x_dbp] = 0.0
            x_sbp[x_sbp!=x_sbp] = 0.0
            x_o2sat[x_o2sat!=x_o2sat] = 0.0
            x_resparate[x_resparate!=x_resparate] = 0.0
            x_heartrate[x_heartrate!=x_heartrate] = 0.0

            x_dbp.astype(np.float32)
            x_sbp.astype(np.float32)
            x_o2sat.astype(np.float32)
            x_resparate.astype(np.float32)
            x_heartrate.astype(np.float32)
            x_numerical.astype(np.float32)

            """
            np.nan_to_num(x_dbp, copy=False)
            np.nan_to_num(x_sbp, copy=False)
            np.nan_to_num(x_o2sat, copy=False)
            np.nan_to_num(x_resparate, copy=False)
            np.nan_to_num(x_heartrate, copy=False)
            """
            
            x_dbp = torch.from_numpy(x_dbp)
            x_sbp = torch.from_numpy(x_sbp)
            x_o2sat = torch.from_numpy(x_o2sat)
            x_resparate = torch.from_numpy(x_resparate)
            x_heartrate = torch.from_numpy(x_heartrate)
            x_numerical = torch.from_numpy(x_numerical)
            x_CC_token_input_ids = torch.Tensor(x_CC_token['input_ids'])
            x_CC_token_attention_mask = torch.Tensor(x_CC_token['attention_mask'])
            x_CC_token_token_type_ids = torch.Tensor(x_CC_token['token_type_ids'])


            y = torch.from_numpy(y)
            
            x_dbp = torch.unsqueeze(x_dbp,0)
            x_sbp = torch.unsqueeze(x_sbp,0)
            x_o2sat = torch.unsqueeze(x_o2sat,0)
            x_resparate = torch.unsqueeze(x_resparate,0)
            x_heartrate = torch.unsqueeze(x_heartrate,0)
            x_numerical = torch.unsqueeze(x_numerical,0)

            #y = torch.unsqueeze(y , 1)

            return x_CC_token_input_ids, x_CC_token_attention_mask, x_CC_token_token_type_ids, x_dbp, x_sbp, x_o2sat, x_resparate, x_heartrate, x_numerical, y
        else:
            y = self.mimic_labels.iloc[idx, 1]
            x = self.mimic_labels.iloc[idx, 2:]

            if self.transform:
                x = self.transform(x)

            
            
            x = np.array(x)
            y = np.array(y)
            
            

            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
            
            x = torch.unsqueeze(x,0)
            #y = torch.unsqueeze(y , 1)

            return x, y

