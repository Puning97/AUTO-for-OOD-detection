import torch,os
from model_pre.model_loader import get_model
import torch.backends.cudnn as cudnn
from seed import set_seed
from data_pre.online_loader import set2loader
from utils.threshold_tool import threshold_tool
from model_pre.resnet import resnet50
import config as cfg
set_seed(cfg.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

train_loader,val_loader,in_out_dataloader,in_clas=set2loader(cfg.in_dataset,cfg.out_dataset)
if cfg.model_pretrain=='resnet50':
    model1=get_model(in_clas ,True,'imagenet',cfg.model_pretrain)
mean_th,std_th=threshold_tool(model1,train_loader,device)