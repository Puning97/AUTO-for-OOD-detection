from utils.memory_tool import memory_gen,memory_rand_gen
import torch,os
from model_pre.model_loader import get_model
import torch.backends.cudnn as cudnn
from utils.online_train_mem import online_training,online_training_odin
from seed import set_seed
from data_pre.online_loader import set2loader
from utils.print_results import get_and_print_results
from model_pre.trans_model import pre_model
import config as cfg
set_seed(cfg.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

train_loader,val_loader,in_out_dataloader,in_clas=set2loader(cfg.in_dataset,cfg.out_dataset)

mem_bank,label_mem=memory_rand_gen(train_loader,100)

if cfg.model_pretrain=='resnet50':
    model1=get_model(in_clas ,True,'imagenet',cfg.model_pretrain)
    model2=get_model(in_clas ,True,'imagenet',cfg.model_pretrain)
else:
    model1=pre_model(cfg.model_pretrain)
    model2=pre_model(cfg.model_pretrain)
model1 = torch.nn.DataParallel(model1)
model2 = torch.nn.DataParallel(model2)
model1.cuda()
model2.cuda()
#mean_th,std_th=threshold_tool(model1,train_loader,device)


if cfg.model_pretrain=='resnet50':
    mean_th=0.9369390606880188
    std_th=0.1566126048564911
if cfg.model_pretrain=='Vit_b_16':
    mean_th=0.8033975958824158
    std_th=0.24346405267715454
if cfg.ood_score in ['msp','energy']:
    in_score,out_score=online_training(cfg.ood_score,model1,model2,in_out_dataloader,device,mean_th,std_th,mem_bank,label_mem)
    print(len(in_score),len(out_score))
    get_and_print_results(in_score, out_score)
elif cfg.ood_score =='odin':
    in_score,out_score=online_training_odin(cfg.ood_score,model1,model2,in_out_dataloader,device,mean_th,std_th,mem_bank,label_mem,cfg.T,cfg.noise)
    get_and_print_results(in_score, out_score)
