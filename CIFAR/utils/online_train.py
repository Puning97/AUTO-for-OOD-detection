from __future__ import print_function
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import config as cfg
import torch.nn as nn
from ood_tool.optim_tool import part_opti
from ood_tool.odin_tool import ODIN

def batch_train(in_out_idx,net,input,optim):
    if in_out_idx==0:
        optim.zero_grad()
        loss_func=nn.CrossEntropyLoss()
        logit_in = net(input)
        loss = loss_func(logit_in, torch.argmax(logit_in, dim=1))
        loss.backward()
        optim.step()
    elif in_out_idx==1:
        optim.zero_grad()
        loss_func =nn.BCEWithLogitsLoss()
        logit_out = net(input)
        loss = loss_func(logit_out,
                           torch.full([logit_out.shape[0], logit_out.shape[1]], 1 / logit_out.shape[1]).cuda())
        loss.backward()
        optim.step()

def online_training(score,net,loader,device,mean,std,Tem=1.0):
    net.eval()
    in_dis_score = []
    out_dis_score = []
    right_score = []
    wrong_score = []
    in_training_count = 0
    out_training_count = 0
    wrong_train_in = 0
    wrong_train_out = 0
    in_border=0
    out_border=0
    border_outcount=[]
    print(score)

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    print('In-dis-part: ', cfg.in_dis_train)
    print('OOD part: ', cfg.out_dis_train)

    parameters = list(net.parameters())
    if cfg.opti_part == 'all':
        optimizer = torch.optim.SGD(parameters, lr=0.001, momentum=0., weight_decay=0.)
    else:
        optimizer = part_opti(net)

    for batch_idx, (data, dis_idx) in enumerate(loader):
        image, label = data[0], data[1]
        # if batch_idx >train_start+1:
        # break
        if batch_idx==0:
            in_border=mean
            out_border=mean-cfg.hyperpara*std
            border_outcount.append(out_border)

        image = image.to(device)
        logits = net(image)
        smax = to_np(F.softmax(logits, dim=1))

        pre_max = np.max(smax, axis=1)
        #print(out_border)
        if pre_max<out_border:
            out_training_count+=1
            border_outcount.append(pre_max)
            #print(border_outcount)
            out_border = np.array(border_outcount).mean()
            batch_train(1,net,image,optimizer)
            if dis_idx==0:
                wrong_train_out+=1
        elif pre_max>in_border:
            in_training_count+=1
            batch_train(0, net, image, optimizer)
            if dis_idx==1:
                wrong_train_in+=1

        preds = np.argmax(smax, axis=1)
        label = label.numpy().squeeze()
        if score == 'energy':
            all_score = -to_np(Tem * torch.logsumexp(logits / Tem, dim=1))
        elif score == 'msp':
            all_score = -np.max(to_np(F.softmax(logits / Tem, dim=1)), axis=1)
        if dis_idx == 0:  # in_distribution
            in_dis_score.append(all_score)
            if preds == label:
                right_score.append(all_score)
            else:
                wrong_score.append(all_score)
        else:  # ood ditribution
            out_dis_score.append(all_score)
        if (batch_idx%1000==0)&(batch_idx>0):
            print(batch_idx)           
            print('Wrong Train Example: In-dis: {}       OOD: {}'.format(wrong_train_in, wrong_train_out))
    # get_and_print_results(np.array(in_dis_score).copy(), np.array(out_dis_score).copy())
            print('Training Example Count: In-disribution: {}       OOD: {}'.format(in_training_count, out_training_count))
    # print('In-distribution Accuracy: {:.2f}%'.format(100*len(right_score)/(len(right_score)+len(wrong_score))))
            print('In-distribution Accuracy: {:.2f}%'.format(100 * len(right_score) / (len(right_score) + len(wrong_score))))

    return np.array(in_dis_score).copy(), np.array(out_dis_score).copy()


def online_training_odin(score,net,loader,device,mean,std,Tem=1.0,noise=0.):
    net.eval()
    in_dis_score = []
    out_dis_score = []
    right_score = []
    wrong_score = []
    in_training_count = 0
    out_training_count = 0
    wrong_train_in = 0
    wrong_train_out = 0
    in_border=0
    out_border=0
    border_outcount=[]
    border_incount=[]
    print(score)

    #concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    print('In-dis-part: ', cfg.in_dis_train)
    print('OOD part: ', cfg.out_dis_train)

    parameters = list(net.parameters())
    if cfg.opti_part == 'all':
        optimizer = torch.optim.SGD(parameters, lr=0.001, momentum=0., weight_decay=0.)
    else:
        optimizer = part_opti(net)

    for batch_idx, (data, dis_idx) in enumerate(loader):
        image, label = data[0], data[1]
        # if batch_idx >train_start+1:
        # break
        if batch_idx==0:
            in_border=mean
            out_border=mean-cfg.hyperpara*std
            border_outcount.append(out_border)
            border_incount.append(in_border)

        image = image.to(device)

        image = Variable(image, requires_grad=True)
        logits = net(image)
        smax = to_np(F.softmax(logits, dim=1))
        odin_score = ODIN(image, logits, net, Tem, noise, device)
        all_score = -np.max(odin_score, 1)
        image = Variable(image, requires_grad=False)

        pre_max = np.max(smax, axis=1)
        #print(out_border)
        if pre_max<out_border:
            out_training_count+=1
            border_outcount.append(pre_max)
            out_border=np.array(border_outcount).mean()
            batch_train(1,net,image,optimizer)
            if dis_idx==0:
                wrong_train_out+=1
        elif pre_max>in_border:
            in_training_count+=1
            border_incount.append(pre_max)
            in_border = np.array(border_incount).mean()
            batch_train(0, net, image, optimizer)
            if dis_idx==1:
                wrong_train_in+=1

        preds = np.argmax(smax, axis=1)
        label = label.numpy().squeeze()

        if dis_idx == 0:  # in_distribution
            in_dis_score.append(all_score)
            if preds == label:
                right_score.append(all_score)
            else:
                wrong_score.append(all_score)
        else:  # ood ditribution
            out_dis_score.append(all_score)
        if (batch_idx % 1000 == 0):
            print(batch_idx)
    print('Wrong Train Example: In-dis: {}       OOD: {}'.format(wrong_train_in, wrong_train_out))
    # get_and_print_results(np.array(in_dis_score).copy(), np.array(out_dis_score).copy())
    print('Training Example Count: In-disribution: {}       OOD: {}'.format(in_training_count, out_training_count))
    # print('In-distribution Accuracy: {:.2f}%'.format(100*len(right_score)/(len(right_score)+len(wrong_score))))
    print('In-distribution Accuracy: {:.2f}%'.format(100 * len(right_score) / (len(right_score) + len(wrong_score))))

    return np.array(in_dis_score).copy(), np.array(out_dis_score).copy()
