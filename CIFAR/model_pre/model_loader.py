import torch
import config as cfg
def get_model(num_classes, load_ckpt=False,dataset='',model_arch=''):
    if dataset == 'imagenet':
        if model_arch == 'resnet18':
            from model_pre.resnet import resnet18
            model = resnet18(num_classes=num_classes, pretrained=True)
        elif model_arch == 'resnet50':
            from model_pre.resnet import resnet50
            model = resnet50(num_classes=num_classes, pretrained=True)
        elif model_arch == 'mobilenet':
            from model_pre.mobilenet import mobilenet_v2
            model = mobilenet_v2(num_classes=num_classes, pretrained=True)
    else:
        if model_arch == 'resnet18':
            from model_pre.resnet import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes)
        elif model_arch in ['resnet34','resnet34_logitnorm']:
            from model_pre.resnet import resnet34_cifar
            model = resnet34_cifar(num_classes=num_classes)
        elif model_arch == 'resnet50':
            from model_pre.resnet import resnet50
            model = resnet50(num_classes=num_classes)
        elif model_arch=='wrn':
            from model_pre.wrn import wrn40_2
            model = wrn40_2(num_classes=num_classes)
        elif model_arch=='densenet':
            from model_pre.densenet import DenseNet3
            model = DenseNet3(100, num_classes, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=None)
        else:
            assert False, 'Not supported model arch: {}'.format(model_arch)

        if load_ckpt:
            ckpt_name=cfg.ckpt_name
            print(ckpt_name)
            checkpoint = torch.load(ckpt_name)
            model.load_state_dict(checkpoint['net'])
    model.cuda()
    model.eval()
    # get the number of model parameters
    #print('Number of model parameters: {}'.format(
        #sum([p.data.nelement() for p in model.parameters()])))
    return model