

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

from dataloader import MimicLoader
from model import TestModel, TestModel2
from model_maac import MAAC, encoder

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

from copy import deepcopy

import datetime

import torchsummary





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

    mimicTrain = MimicLoader(
        data_CC=data_CC, 
        data_Seq=data_Seq, 
        annotations_file=data, 
        transform=train_transformer, 
        train=True)

    mimicTest = MimicLoader(
        data_CC=data_CC, 
        data_Seq=data_Seq,
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
    
   
    
    if modelNum == '1':
        model = TestModel()
    elif modelNum == '2':
        model = TestModel2()
    elif modelNum == 'macc':
        encoderModel = encoder(encoder_pretrained=True)
        #print(encoderModel)
        model = MAAC(encoderModel=encoderModel)
        

    #print(model)

    NGPU = torch.cuda.device_count()
    device = torch.device("cuda")

    model = nn.DataParallel(model)   # 4개의 GPU를 이용할 경우 pre_trained

    print("-------------------------")
    for i in range(NGPU):
        print(torch.cuda.get_device_name(i))
    print(sys.version)
    print("-------------------------")

    model.to(device)

    #print(model)


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
   

    for epoch in range(num_epoch):
        model.train()

        if data_CC == False and data_Seq == False:

            for idx, (xx, label) in enumerate(train_loader):
                x = xx.to(device)
                x = x.float()
                #label = label.float()
                # label = list(label)
                y_ = label.to(device)
                
                #print(x.shape)

                # train데이터 셋 feedforwd 과정
                output = model.forward(x)

                #print(output.shape, y_.shape)

                # loss 계산
                loss = loss_func(output, y_)

                # optimizer 초기화 및 weight 업데이트
                optimizer.zero_grad()  # 그래디언트 제로로 만들어주는 과정
                #f = loss.mean()
                loss.backward()  # backpropagation
                
                #loss.mean().backward()
                optimizer.step()


                #if idx % 100 == 0:

        elif data_CC ==True and data_Seq == True:
            for idx, (
                x_CC_token_input_ids, 
                x_CC_token_attention_mask, 
                x_CC_token_token_type_ids, 
                x_dbp, 
                x_sbp, 
                x_o2sat, 
                x_resparate, 
                x_heartrate, 
                x_numerical, 
                label
                ) in enumerate(train_loader):
                """
                print(x_CC_token_input_ids.shape)
                print(x_CC_token_attention_mask.shape)
                print(x_CC_token_token_type_ids.shape)
                print(x_dbp.shape)
                print(x_sbp.shape)
                print(x_o2sat.shape)
                print(x_resparate.shape)
                print(x_heartrate.shape)
                print(x_numerical.shape)
                print(label.shape)
                """

                #print(x_CC_token_input_ids, x_CC_token_attention_mask, x_CC_token_token_type_ids, x_dbp.shape, x_sbp.shape, x_o2sat.shape, x_resparate.shape, x_heartrate.shape, x_numerical.shape, label.shape)
                x_CC_token_input_ids = x_CC_token_input_ids.to(device).long()
                x_CC_token_attention_mask = x_CC_token_attention_mask.to(device).long()
                x_CC_token_token_type_ids = x_CC_token_token_type_ids.to(device).long()
                x_dbp = x_dbp.to(device).to(torch.float32)
                x_sbp = x_sbp.to(device).to(torch.float32)
                x_o2sat = x_o2sat.to(device).to(torch.float32)
                x_resparate = x_resparate.to(device).to(torch.float32)
                x_heartrate = x_heartrate.to(device).to(torch.float32)
                x_numerical = x_numerical.to(device).to(torch.float32)
                y_ = label.to(device)

                
                #print(x.shape)

                # train데이터 셋 feedforwd 과정
                output = model.forward(
                    x_CC_token_input_ids,
                    x_CC_token_attention_mask,
                    x_CC_token_token_type_ids,
                    x_dbp,
                    x_sbp,
                    x_o2sat,
                    x_resparate,
                    x_heartrate,
                    x_numerical
                    )

                #print(output.shape, y_.shape)

                # loss 계산
                loss = loss_func(output, y_)

                # optimizer 초기화 및 weight 업데이트
                optimizer.zero_grad()

                loss.backward()  # backpropagation

                optimizer.step()


        print('epoch : ', epoch)
        #print('loss : ', loss.data)
        

        model.eval()
        test_loss = 0
        correct = 0
        
        loss_func

        guesses = np.array([])
        labels = np.array([])

        with torch.no_grad():
            if data_CC == False and data_Seq == False:
                for idx, (data, target) in enumerate(test_loader):
                    data, target = data.to(torch.device('cuda')), target.to(torch.device('cuda'))
                    #target = target.float()
                    data=data.float()
                    output = model(data)
                    #print(output, target)
                    lossT = loss_func(output, target)
                    test_loss +=  lossT.item()
                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

                    tmp1 = np.array(pred.to('cpu'))
                    tmp2 = np.array(target.to('cpu'))

                    tt1 = np.array(tmp1[:])
                    tt2 = np.array(tmp2[:])

                    guesses = np.append(guesses, tt1)
                    labels = np.append(labels, tt2)

                    #print(len(guesses))
            elif data_CC ==True and data_Seq == True:
                for idx, (
                    x_CC_token_input_ids, 
                    x_CC_token_attention_mask, 
                    x_CC_token_token_type_ids, 
                    x_dbp, 
                    x_sbp, 
                    x_o2sat, 
                    x_resparate, 
                    x_heartrate, 
                    x_numerical, 
                    target
                    ) in enumerate(test_loader):

                    x_CC_token_input_ids = x_CC_token_input_ids.to(device).long()
                    x_CC_token_attention_mask = x_CC_token_attention_mask.to(device).long()
                    x_CC_token_token_type_ids = x_CC_token_token_type_ids.to(device).long() 
                    x_dbp = x_dbp.to(device)
                    x_sbp = x_sbp.to(device)
                    x_o2sat = x_o2sat.to(device)
                    x_resparate = x_resparate.to(device)
                    x_heartrate = x_heartrate.to(device)
                    x_numerical = x_numerical.to(device)
                    y_ = target.to(device)

                    output = model.forward(
                        x_CC_token_input_ids,
                        x_CC_token_attention_mask,
                        x_CC_token_token_type_ids,
                        x_dbp,
                        x_sbp,
                        x_o2sat,
                        x_resparate,
                        x_heartrate,
                        x_numerical
                        )

                    lossT = loss_func(output, target)
                    test_loss +=  lossT.item()
                    pred = output.argmax(dim=1, keepdim=True)

                    tmp1 = np.array(pred.to('cpu'))
                    tmp2 = np.array(target.to('cpu'))

                    tt1 = np.array(tmp1[:])
                    tt2 = np.array(tmp2[:])

                    guesses = np.append(guesses, tt1)
                    labels = np.append(labels, tt2)
                

        #guesses = guesses.astype(int)
        labels = labels.astype(int)
        guesses = guesses.astype(int)

        guesses = list(guesses)
        labels = list(labels)

        #print(guesses,'\n',labels)

        print(classification_report(labels, guesses, labels=[0,1]))

        #misc (acc 계산, etc) 
        acc = accuracy_score(labels, guesses)
        f_score = f1_score(labels, guesses, average='macro')
        
        

        if acc > best_acc:
            best_acc = acc
            best_acc_model = deepcopy(model.state_dict())

            with open('./result/' + result_name + "_" + str(now) + "_best_acc.txt", "w") as text_file:
                print("epoch:", epoch, file=text_file)
                print("test loss:", test_loss, file=text_file)
                print(classification_report(labels, guesses, digits=3), file=text_file)
                print(model, file=text_file)
            
            torch.save(best_acc_model, './result/' + result_name + "_" + str(now) + '_best_acc.pt')

        if f_score > best_f1:
            best_f1 = f_score
            best_f1_model = deepcopy(model.state_dict())
            
            with open('./result/' + result_name + "_" + str(now) + "_best_f1.txt", "w") as text_file:
                print("epoch:", epoch, file=text_file)
                print("test loss:", test_loss, file=text_file)
                print(classification_report(labels, guesses, digits=3), file=text_file)
                print(model, file=text_file)

            torch.save(best_f1_model, './result/' + result_name + "_" + str(now) + '_best_f1.pt')

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

    

    

    trainEffNet(parser)
