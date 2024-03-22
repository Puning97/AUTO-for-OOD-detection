import os
import bisect
import torch
import torchvision
import config as cfg
from torchvision import transforms
import warnings
from torch.utils.data import *
imagesize = 32

batch_size= cfg.test_bs

kwargs = {'num_workers': 2, 'pin_memory': False}

transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def train_loader(in_dataset):
    if in_dataset == "CIFAR-10":
        # Data loading code
        trainset = torchvision.datasets.CIFAR10(root='cifarpy/', train=True, download=False,
                                                  transform=transform_test)
        num_classes = 10
        num_classes = 173
    elif in_dataset == 'Imagenet':
        trainset = torchvision.datasets.ImageFolder(root='imagenet_series/ImageNet/train',
                                                  transform=transform_test_largescale)
        num_classes = 1000
    return trainset

def in_loader(in_dataset):
    if in_dataset == "CIFAR-10":
        # Data loading code
        valset = torchvision.datasets.CIFAR10(root='cifarpy/', train=False, download=False,
                                                  transform=transform_test)
        num_classes = 10
    elif in_dataset == 'Imagenet':
        valset = torchvision.datasets.ImageFolder(root='imagenet_series/ImageNet/val',
                                                  transform=transform_test_largescale)
        num_classes = 1000
    return valset,len(valset),num_classes

def out_loader(val_dataset):
    #cifar
    if val_dataset == 'SVHN':
        from data_pre.svhn_loader import SVHN
        out_set=SVHN('svhn/', split='test', transform=transform_test, download=False)
    elif val_dataset == 'Textures':
        if cfg.in_dataset=='Imagenet':
            transform=transform_test_largescale
        else: transform = transform_test
        out_set=torchvision.datasets.ImageFolder(root="ImageNet_OOD_dataset/dtd/images", transform=transform)
    # Imagenet
    elif val_dataset == 'iNaturalist':
        out_set=torchvision.datasets.ImageFolder("ImageNet_OOD_dataset/iNaturalist",
                                             transform_test_largescale)
    elif val_dataset == 'Places50':
        out_set=torchvision.datasets.ImageFolder("ImageNet_OOD_dataset/Places",
                                             transform_test_largescale)
    elif val_dataset == 'SUN':
        out_set=torchvision.datasets.ImageFolder("ImageNet_OOD_dataset/SUN",
                                             transform_test_largescale)
    else:
        val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("./datasets/ood_data/{}",
                                                                                      transform=transform_test),
                                                     batch_size=batch_size, shuffle=False, num_workers=2)
    return out_set

class Mydataset(ConcatDataset):
    def cumsum(self,sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        #self.in_datasets = in_dataset
        #self.out_datasets = out_dataset
        self.datasets=list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx

    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


def set2loader(in_dataset,out_dataset):
    in_dis_set,length,in_num_classes=in_loader(in_dataset)
    out_dis_set=out_loader(out_dataset)
    train_set=train_loader(in_dataset)

    print('In-distribution Dataset:',in_dataset)
    print("OOD Dataset:",out_dataset)
    #print('Val Dataset:',val_dataset)

    conc_set=Mydataset([in_dis_set,out_dis_set])
    #val_set=Mydataset([in_dis_set,val_dis_set])
    length_dataset=len(conc_set)
    #print('Test loader length:{}'.format(length_dataset))
    #_,conc_val_set=torch.utils.data.random_split(val_set,[int((1-cfg.val_perc)*length_dataset),int(cfg.val_perc*length_dataset)],generator=torch.Generator().manual_seed(cfg.seed))

    in_out_loader=torch.utils.data.DataLoader(conc_set,batch_size=1,shuffle=True,num_workers=2)
    #val_loader=torch.utils.data.DataLoader(val_set,batch_size=1,shuffle=True,num_workers=2)
    train_mean_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2)

    return train_mean_loader,'abc',in_out_loader,in_num_classes