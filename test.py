

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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, roc_curve, roc_auc_score


from copy import deepcopy

import datetime

import torchsummary

import matplotlib.pyplot as plt

from model_maac import MAAC

def get_tpr(y_true,y_scores,threshold):
    predict_positive_num = len(y_scores[y_scores >= threshold]) 
    tp = len([x for x in y_true[:predict_positive_num] if x == 1])
    ground_truth = len(y_true[y_true==1]) 
    if ground_truth != 0:
        tpr =  tp / ground_truth
    else:
        tpr = 0
    return tpr
    
def get_fpr(y_true,y_scores,threshold):
    predict_positive_num = len(y_scores[y_scores >= threshold] )
    fp = len([x for x in y_true[:predict_positive_num] if x == 0 ])
    ground_negative = len(y_true[y_true==0])
    if ground_negative != 0:
        fpr = fp / ground_negative
    else:
        fpr = 0
    return fpr

def roc_plot(y_true,y_scores):
    tpr , fpr = [] , []

    for _ in y_scores: # y_scores 를 thresholds 처럼 사용했음
        tpr.append(get_tpr(y_true,y_scores,_ ))
        fpr.append(get_fpr(y_true,y_scores,_ ))

    fig = plt.figure(figsize=(9, 6))

    #3d container
    ax = plt.axes(projection = '3d')
    #3d scatter plot
    ax.plot3D(fpr, y_scores, tpr)
    ax.scatter3D(fpr, y_scores, tpr)
    ax.plot3D([0,1],[1,0],[0,1])
    #give labels
    ax.set_xlabel('False-Positive-Rate')
    ax.set_ylabel('Thresholds')
    ax.set_zlabel('True-Positive-Rate')
    ax.set_title('ROC Curve 3D')
    #set fpr,tpr limit 0 to 1
    ax.set_xlim(0,1)
    ax.set_zlim(0,1)
    ax.view_init(26) #방향 돌려서 보기. 
    plt.savefig('ROC_Curve_3D.png')
    
    fig = plt.figure(figsize = (9,6))
    plt.plot(fpr, tpr)
    plt.scatter(fpr,tpr)
    plt.plot([0,1],[0,1])
    plt.xlabel('False-Positive-Rate')
    plt.ylabel('True-Positive-Rate')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title('ROC Curve 2D')
    plt.savefig('ROC_Curve_2D.png')


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
    #pre_trained = options.
    weight = options.weight
    only_triage = options.only_triage

    if options.only_triage == 'True':

        only_triage = True
    else:
        only_triage = False

    now = datetime.datetime.now()
    
    print("===========================================")
    print("Test Start")
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
            annotations_file=data, 
            transform=train_transformer, 
            train=True)
        mimicTest = MimicLoader_dataset1(
            annotations_file=data, 
            train=False)
    else:
        mimicTrain = MimicLoader_dataset1_onlyTriage(
            annotations_file=data, 
            transform=train_transformer, 
            train=True)
        mimicTest = MimicLoader_dataset1_onlyTriage(
            annotations_file=data, 
            train=False)


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
    """
    if modelNum == '1':
        model = TestModel()
    elif modelNum == '2':
        model = TestModel2()
    """
    encoderModel = encoder(encoder_pretrained=False).to(device)

    
    if only_triage == False:
        encoderModel = encoder(encoder_pretrained=True)
        #print(encoderModel)
        model = MAAC(encoderModel)
    else:
        encoderModel = encoder(encoder_pretrained=True)
        #print(encoderModel)
        model = MAAC_onlyTriage(encoderModel)


    NGPU = torch.cuda.device_count()
    

    model = nn.DataParallel(model)   # 4개의 GPU를 이용할 경우 pre_trained
    model.load_state_dict(torch.load(weight))

    print("-------------------------")
    for i in range(NGPU):
        print(torch.cuda.get_device_name(i))
    print(sys.version)
    print("-------------------------")

    model.to(device)


    loss_func = None
    if loss_function == 'criterion':
        loss_func = nn.CrossEntropyLoss()  # 크로스엔트로피 loss 객체, softmax를 포함함
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
   

        

    model.eval()
    test_loss = 0
    correct = 0
    
    loss_func

    guesses = np.array([])
    labels = np.array([])

    with torch.no_grad():
        if only_triage == False:

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

                #print(len(guesses))
        else:
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
            

    #guesses = guesses.astype(int)
    guesses = list(guesses)
    total = 0
    for g in guesses:
        total += g

    print(guesses[:100])
    print(labels[:100])
    labels = [int(l) for l in labels]
    total /= len(guesses)
    #print(guesses)
    #guesses = [0 if guesses[i] <= 0.2 else 1 for i in range(len(guesses))]
    #labels =  [0 if labels[i] <= 0.2 else 1 for i in range(len(labels))]
    
    #print(guesses,'\n',labels)

    #print(classification_report(labels, guesses, labels=[0,1]))

    #print('roc auc value {}'.format(roc_auc_score(labels,guesses)))

    #misc (acc 계산, etc) 
    #acc = accuracy_score(labels, guesses)
    #f_score = f1_score(labels, guesses, average='macro')


    #roc_plot(np.array(guesses), np.array(labels))

    print('roc auc value {}'.format(roc_auc_score(np.array(labels),np.array(guesses))))

    fprs, tprs, thresholds = roc_curve(np.array(labels), np.array(guesses))
    
    # 대각선
    plt.plot([0,1],[0,1],label='STR')

    # ROC
    plt.plot(fprs,tprs,label='ROC')


    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid()
    plt.savefig('roc.png')
    
        

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
    parser.add_option("--weight", "-W", default="", dest="weight", type=str)
    parser.add_option("--only_triage", default="False", dest="only_triage", type=str)

    

    trainEffNet(parser)
