from __future__ import print_function
import torch
import os
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import config as cfg
import torch.nn as nn
from ood_tool.optim_tool import part_opti
from ood_tool.odin_tool import ODIN
import time
from utils.consistent_loss import get_margin_loss
from utils.plot_util import plot_distribution
#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def batch_train(mem_b,mem_l,net,net2,input,optim):
    to_np = lambda x: x.data.cpu().numpy()
    x1=0
    while x1<cfg.train_time:
        optim.zero_grad()
    
        #if cfg.in_dataset == 'Imagenet':
            #loss_func1 = nn.CrossEntropyLoss(reduction='sum')
        #else:
        loss_func1 = nn.CrossEntropyLoss()

        if cfg.in_dataset=='Imagenet':
            logit_in = net(mem_b.cuda())
            loss1=loss_func1(logit_in.cpu(), mem_l)
            #first_idx=True
            #for i in range (len(mem_b)):
                #print(mem_b[i].shape)
                #logit_in = net(mem_b[i].cuda())
                #mem_b[i].cpu()
                #print(mem_b[i].device)
                #if first_idx==True:
                    #loss1 = loss_func1(logit_in.cpu(), mem_l[i])
                    #first_idx = False
                #else:
                    #loss1 += loss_func1(logit_in.cpu(), mem_l[i])
                #mem_l[i].cpu()
                
            #loss1=loss1/1000.
    
        #loss_func2 =nn.BCEWithLogitsLoss()
        logit_out = net(input)
        #logit_out_org=net2(input)
        #smax_out_org = to_np(F.softmax(logit_out_org, dim=1))
    #print(smax_out_org)
        #pred_out_org = np.argmax(smax_out_org, axis=1)
    #print(pred_out_org)
        loss2 = 1.*-(logit_out.mean(1) - torch.logsumexp(logit_out, dim=1)).mean()
        #loss3=get_margin_loss(logit_out.cpu(),pred_out_org,cfg.consis_idx)
        #print(loss1,loss2,loss3)
    #loss=cfg.in_weight*loss1+cfg.ood_weight*loss2#+0.2*loss3
        loss=cfg.ood_weight*loss2+cfg.in_weight*loss1#+cfg.consis_weight*loss3
        loss.backward()
        optim.step()
        x1+=1
    
def online_training(score,net,net2,loader,device,mean,std,mem_bank,mem_label,Tem=1.0):
    train_idx=0
    start_time=time.time()
    net.eval()
    net2.eval()
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
    mem_idx=0
    part_in=[]
    part_out=[]
    print(score)
    print('In_weight:  {}      Out_weight:{}    Consis_weight:{}'.format(cfg.in_weight,cfg.ood_weight,cfg.consis_weight))
    print('Consistent threshold:{}'.format(cfg.consis_idx))
    print('In_border:  {}      Out_border:{}'.format(cfg.hyperpara_in,cfg.hyperpara_out))
    print('Online Train:  {}'.format(cfg.online_train))
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    parameters = list(net.parameters())
    if cfg.opti_part == 'all':
        optimizer = torch.optim.SGD(parameters, lr=0.001, momentum=0., weight_decay=0.)
    else:
        optimizer = part_opti(net)

    for batch_idx, (data, dis_idx) in enumerate(loader):
        image, label = data[0], data[1]
        if batch_idx==0:
            in_border=mean+cfg.hyperpara_in*std
            out_border=mean-cfg.hyperpara_out*std
            border_outcount.append(out_border)

        image = image.to(device)
        logits = net(image)
        smax = to_np(F.softmax(logits, dim=1))
        pre_max = np.max(smax, axis=1)


        if pre_max<out_border:
            out_training_count+=1
            border_outcount.append(pre_max)
            out_border = np.array(border_outcount).mean()
            if cfg.online_train:
                batch_train(mem_bank,mem_label,net,net2,image,optimizer)
            if dis_idx==0:
                wrong_train_out+=1
            train_idx+=1

        elif pre_max>in_border:
            pred=np.argmax(smax,axis=1)
            #if cfg.in_dataset in ['CIFAR-10','CIFAR-100']:
                #for i in range (len(smax)):
                    #if pred==mem_label[i].item():
                        #mem_bank[i]=torch.squeeze(image,dim=0)
            if mem_idx<len(mem_label):
                mem_bank[mem_idx]=torch.squeeze(image,dim=0)
                mem_label[mem_idx]=torch.Tensor(pred)
                mem_idx+=1
            else:
                mem_idx=0
                mem_bank[mem_idx] = torch.squeeze(image, dim=0)
                mem_label[mem_idx] = torch.Tensor(pred)
            #else:
                #for i in range (len(mem_label)):
                    #for j in range (len(mem_label[i])):
                        #if pred==mem_label[i][j].item():
                            #mem_bank[i][j] = torch.squeeze(image, dim=0)
            in_training_count+=1

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
            part_in.append(all_score)
            if preds == label:
                right_score.append(all_score)
            else:
                wrong_score.append(all_score)
        else:  # ood ditribution
            out_dis_score.append(all_score)
            part_out.append(all_score)
        if (batch_idx%1000==0)&(batch_idx>0):
            print(batch_idx)
            working_time=time.time()-start_time
            #print("Training Time: ", working_time)
            print('Wrong Train Example: In-dis: {}       OOD: {}'.format(wrong_train_in, wrong_train_out))
            print('Training Example Count: In-disribution: {}       OOD: {}'.format(in_training_count, out_training_count))
            print('In-distribution Accuracy: {:.2f}%'.format(100 * len(right_score) / (len(right_score) + len(wrong_score))))
        if (batch_idx%3000==0)&(batch_idx>0):
            plot_distribution(str(batch_idx), part_in, part_out, cfg.out_dataset)    
            part_in=[]
            part_out=[]
    print("----------------------")
    print("Training Time: ", working_time)
    print('Training Example Count: In-disribution: {}       OOD: {}'.format(in_training_count, out_training_count))
    print('In-distribution Accuracy: {:.2f}%'.format(100 * len(right_score) / (len(right_score) + len(wrong_score))))
    print('Saving..')
    state = {
            'net': net.state_dict(),
            }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if cfg.online_train:
        torch.save(state, './checkpoint/after_training_'+cfg.in_dataset+'_'+cfg.out_dataset+'_'+cfg.model_pretrain+'.pth.tar')
    
    return np.array(in_dis_score).copy(), np.array(out_dis_score).copy()


def online_training_odin(score,net,net2,loader,device,mean,std,mem_bank,mem_label,Tem=1.0,noise=0.):
    train_idx = 0
    net.eval()
    net2.eval()
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
    print('In_weight:  {}      Out_weight:{}    Consis_weight:{}'.format(cfg.in_weight, cfg.ood_weight,
                                                                         cfg.consis_weight))
    print('Consistent threshold:{}'.format(cfg.consis_idx))
    print('In_border:  {}      Out_border:{}'.format(cfg.hyperpara_in, cfg.hyperpara_out))
    print('Online Train:  {}'.format(cfg.online_train))

    #concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

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
            in_border=mean+cfg.hyperpara_in*std
            out_border=mean-cfg.hyperpara_out*std
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
            out_border = np.array(border_outcount).mean()
            if cfg.online_train:
                batch_train(mem_bank,mem_label,net,net2,image,optimizer)
            if dis_idx==0:
                wrong_train_out+=1
            train_idx+=1

        elif pre_max>in_border:
            pred=np.argmax(smax,axis=1)
            if cfg.in_dataset in ['CIFAR-10','CIFAR-100']:
                for i in range (len(smax)):
                    if pred==mem_label[i].item():
                        mem_bank[i]=torch.squeeze(image,dim=0)
                #if mem_idx<len(mem_label):
                    #mem_bank[mem_idx]=torch.squeeze(image,dim=0)
                    #mem_label[mem_idx]=torch.Tensor(pred)
                    #mem_idx+=1
                #else:
                    #mem_idx=0
                    #mem_bank[mem_idx] = torch.squeeze(image, dim=0)
                    #mem_label[mem_idx] = torch.Tensor(pred)
            else:
                for i in range (len(mem_label)):
                    for j in range (len(mem_label[i])):
                        if pred==mem_label[i][j].item():
                            mem_bank[i][j] = torch.squeeze(image, dim=0)
            in_training_count+=1

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
        if (batch_idx % 1000 == 0)&(batch_idx>0):
            print(batch_idx)
            print('Wrong Train Example: In-dis: {}       OOD: {}'.format(wrong_train_in, wrong_train_out))
    # get_and_print_results(np.array(in_dis_score).copy(), np.array(out_dis_score).copy())
            print('Training Example Count: In-disribution: {}       OOD: {}'.format(in_training_count, out_training_count))
    # print('In-distribution Accuracy: {:.2f}%'.format(100*len(right_score)/(len(right_score)+len(wrong_score))))
            print('In-distribution Accuracy: {:.2f}%'.format(100 * len(right_score) / (len(right_score) + len(wrong_score))))
    print("----------------------")
    print('In-distribution Accuracy: {:.2f}%'.format(100 * len(right_score) / (len(right_score) + len(wrong_score))))

    return np.array(in_dis_score).copy(), np.array(out_dis_score).copy()
