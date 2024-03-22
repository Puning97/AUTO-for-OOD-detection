import torch
import config as cfg
def get_model(num_classes, load_ckpt=False,dataset='',model_arch=''):
    if dataset == 'imagenet':
        if model_arch == 'resnet18':
            from model_pre.resnet import resnet18
            model = resnet18(num_classes=num_classes, pretrained=True)
        elif model_arch == 'resnet50':
            from model_pre.resnet import resnet50,resnet50_react
            if cfg.react is True:
                model = resnet50_react(num_classes=num_classes, pretrained=True)
            else:
                model = resnet50(num_classes=num_classes, pretrained=True)
        elif model_arch == 'mobilenet':
            from model_pre.mobilenet import mobilenet_v2
            model = mobilenet_v2(num_classes=num_classes, pretrained=True)
    model.cuda()
    model.eval()
    # get the number of model parameters
    #print('Number of model parameters: {}'.format(
        #sum([p.data.nelement() for p in model.parameters()])))
    return model