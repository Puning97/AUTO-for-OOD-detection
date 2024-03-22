import config as cfg
import torch
def memory_gen(loader,num_class):
    sample=[]
    first_one=True
    for batch_idx, (data,label) in enumerate(loader):
        for i in range (len(label)):
            if label[i] not in sample:
                label_=torch.tensor([label[i].item()])
                if first_one==True:
                    memory_bank=torch.unsqueeze(data[i], dim=0)
                    label_mem=label_
                    first_one=False
                else:
                    memory_bank=torch.cat([memory_bank,torch.unsqueeze(data[i], dim=0)],dim=0)
                    label_mem=torch.cat([label_mem,label_],dim=0)
                sample.append(label[i])
                if len(sample)==num_class:
                    if cfg.in_dataset=='Imagenet':
                        memory_bank=torch.split(memory_bank,cfg.test_bs,0)
                        label_mem=torch.split(label_mem,cfg.test_bs,0)
                    return memory_bank,label_mem
    
    
def memory_rand_gen(loader,num_class):
    sample=0
    first_one=True
    for batch_idx, (data,label) in enumerate(loader):
        for i in range (len(label)):
            label_=torch.tensor([label[i].item()])
            if first_one==True:
                memory_bank=torch.unsqueeze(data[i], dim=0)
                label_mem=label_
                first_one=False
            else:
                memory_bank=torch.cat([memory_bank,torch.unsqueeze(data[i], dim=0)],dim=0)
                label_mem=torch.cat([label_mem,label_],dim=0)
            sample+=1
            if sample==num_class:
                return memory_bank,label_mem
    