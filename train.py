

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

from dataloader import MimicLoader, MimicLoader_dataset1, MimicLoader_dataset1_onlyTriage
from model import TestModel, TestModel2
from model_maac import MAAC, encoder, MACCwithTransformer, MAAC_onlyTriage

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

from copy import deepcopy

import datetime
from pytorchtools import EarlyStopping
import torchsummary

import wandb

import matplotlib.pyplot as plt



def trainEffNet(parser):

    (options, args) = parser.parse_args()

    batch_size = options.batch_size
    learning_rate = options.learning_rate
    num_epoch = options.num_epoch
    modelNum = options.model
    #class_num = options.class_num
    data = options.data
    result_name = options.result_name
    loss_function = options.loss_function
    save_dir = options.save_dir

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

    wandb.init()

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
        mimicTrain = MimicLoader_dataset1(
            data_CC=data_CC, 
            data_Seq=data_Seq, 
            annotations_file=data, 
            transform=train_transformer, 
            train=True)
        mimicTest = MimicLoader_dataset1(
            data_CC=data_CC, 
            data_Seq=data_Seq,
            annotations_file=data, 
            train=False)
    else:
        mimicTrain = MimicLoader_dataset1_onlyTriage(
            data_CC=data_CC, 
            data_Seq=data_Seq, 
            annotations_file=data, 
            transform=train_transformer, 
            train=True)
        mimicTest = MimicLoader_dataset1_onlyTriage(
            data_CC=data_CC, 
            data_Seq=data_Seq,
            annotations_file=data, 
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
    test_loader = DataLoader(mimicTest,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=options.workers,
                             drop_last=False,
                             pin_memory=True
                             )
    device = torch.device("cuda")
   
    
    
    if modelNum == 'macc':
        if only_triage == False:
            encoderModel = encoder(encoder_pretrained=True)
            #print(encoderModel)
            model = MAAC(encoderModel)
        else:
            encoderModel = encoder(encoder_pretrained=True)
            #print(encoderModel)
            model = MAAC_onlyTriage(encoderModel)

    elif modelNum == 'maccwithTransformer':
        encoderModel = encoder(encoder_pretrained=True)
        model = MACCwithTransformer(encoderModel)
        

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
                Platelet, 
                Potassium, 
                Sodium, 
                Urea_Nitrogen, 
                white_blood_cell, 
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
                Platelet = Platelet.to(device).float()
                Potassium = Potassium.to(device).float()
                Sodium = Sodium.to(device).float()
                Urea_Nitrogen = Urea_Nitrogen.to(device).float()
                white_blood_cell = white_blood_cell.to(device).float()
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
                    Platelet, 
                    Potassium, 
                    Sodium, 
                    Urea_Nitrogen, 
                    white_blood_cell, 
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
                x_acuity, y) in enumerate(train_loader):

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
                    x_acuity)

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
                    Platelet, 
                    Potassium, 
                    Sodium, 
                    Urea_Nitrogen, 
                    white_blood_cell, 
                    pCO2, 
                    pH, 
                    Bilirubin, 
                    x_numerical2, 
                    label) in enumerate(test_loader):

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
                    Platelet = Platelet.to(device).float()
                    Potassium = Potassium.to(device).float()
                    Sodium = Sodium.to(device).float()
                    Urea_Nitrogen = Urea_Nitrogen.to(device).float()
                    white_blood_cell = white_blood_cell.to(device).float()
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
                        Platelet, 
                        Potassium, 
                        Sodium, 
                        Urea_Nitrogen, 
                        white_blood_cell, 
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
                    x_acuity, label) in enumerate(test_loader):

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
                        x_acuity)

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

        with open(save_dir + '/' + result_name + "_" + str(now) + "_epoch" + str(epoch) + ".txt", "w") as text_file:
            print("epoch:", epoch, file=text_file)
            print("test loss:", test_loss, file=text_file)
            print(classification_report(labels, guesses, digits=3), file=text_file)
            print(model, file=text_file)
            print("average", total, file=text_file)
        
        torch.save(best_acc_model, save_dir + '/' + result_name + "_" + str(now) + "_epoch" + str(epoch) + '.pt')

        #misc (acc 계산, etc) 

        if acc > best_acc:
            best_acc = acc
            best_acc_model = deepcopy(model.state_dict())


            with open( save_dir + '/' + result_name + "_" + str(now) + "_best_acc.txt", "w") as text_file:
                print("epoch:", epoch, file=text_file)
                print("test loss:", test_loss, file=text_file)
                print(classification_report(labels, guesses, digits=3), file=text_file)
                print(model, file=text_file)
                print("average", total, file=text_file)
            
            torch.save(best_acc_model,  save_dir + '/' + result_name + "_" + str(now) + '_best_acc.pt')

        if f_score > best_f1:
            best_f1 = f_score
            best_f1_model = deepcopy(model.state_dict())

            
            with open( save_dir + '/' + result_name + "_" + str(now) + "_best_f1.txt", "w") as text_file:
                print("epoch:", epoch, file=text_file)
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
    parser.add_option("--result_name", "-n", default="", dest="result_name", type=str)
    parser.add_option("--loss", default="criterion", dest="loss_function", type=str)
    #parser.add_option("--pre_trained", default=None, dest="pre_trained", type=int)
    #parser.add_option("--weight", "-W", default="", dest="weight", type=str)
    parser.add_option("--data_CC", default="False", dest="data_CC", type=str)
    parser.add_option("--data_Seq", default="False", dest="data_Seq", type=str)
    parser.add_option("--only_triage", default="False", dest="only_triage", type=str)
    parser.add_option("--save_dir", default="./result", dest="save_dir", type=str)
    trainEffNet(parser)
