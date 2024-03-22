#model_cifar
#model_arch='resnet18'
#model_imagenet
model_pretrain='resnet50'
#['Swin_l','Swin_t','Swin_s','Swin_b','Vit_t_16','Vit_s_16','Vit_s_32','Vit_b_16','Vit_b_32','Vit_l_16','Bit_1','Bit_3','resnet50']

#test
seed=1
gpu='4'

#是否将OOD样本作为正样本
out_as_pos=False
optimal_t=False

#ODIN
T=1000
noise=0.005

online_train=True
#dataset
test_bs=16
opti_part='layer4'
opti_part2='bn'

in_dataset="Imagenet"   #"Imagenet"
out_dataset='SUN' #'Textures','iNaturalist','Places50','SUN'

ood_score='energy' # msp odin energy
train_time=1
in_weight=1.
ood_weight=0.25
consis_weight=0.1

consis_idx=0.005
hyperpara_out=3.
hyperpara_in=0.

react=False
if in_dataset=="Imagenet":
    out_datasets = ['Textures','iNaturalist','Places50','SUN']
    model_base=['Vit_b_16','Vit_l_16','Bit_1','Bit_3','Swin_b','Swin_l']
    T_me=1.0
