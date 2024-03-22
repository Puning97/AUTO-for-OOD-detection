import timm
import torch


def pre_model(model_name):
    print(model_name)
    if model_name=='Vit_t_16':
        ViT_t_16=timm.create_model('vit_tiny_patch16_224',pretrained=True)
        return ViT_t_16
    elif model_name=='Vit_s_16':
        ViT_s_16=timm.create_model('vit_small_patch16_224',pretrained=True)
        return ViT_s_16
    elif model_name=='Vit_s_32':
        ViT_s_32=timm.create_model('vit_small_patch32_224',pretrained=True)
        return ViT_s_32
    elif model_name == 'Vit_b_16':
        ViT_b_16=timm.create_model('vit_base_patch16_224',pretrained=True)
        return ViT_b_16
    elif model_name == 'Vit_b_32':
        ViT_b_32=timm.create_model('vit_base_patch32_224',pretrained=True)
        return ViT_b_32
    elif model_name == 'Vit_l_16':
        ViT_l_16=timm.create_model('vit_large_patch16_224',pretrained=True)
        return ViT_l_16
    elif model_name == 'Bit_1':
        BiT_1=timm.create_model('resnetv2_50x1_bitm',pretrained=True)
        return BiT_1
    elif model_name == 'Bit_3':
        BiT_3=timm.create_model('resnetv2_50x3_bitm',pretrained=True)
        return BiT_3
    elif model_name == 'Swin_t':
        Swin_t=timm.create_model('swin_tiny_patch4_window7_224',pretrained=True)
        return Swin_t
    elif model_name == 'Swin_s':
        Swin_s=timm.create_model('swin_small_patch4_window7_224',pretrained=True)
        return Swin_s
    elif model_name == 'Swin_b':
        Swin_b=timm.create_model('swin_base_patch4_window7_224',pretrained=True)
        return Swin_b
    elif model_name == 'Swin_l':
        Swin_l=timm.create_model('swin_large_patch4_window7_224',pretrained=True)
        return Swin_l
    elif model_name == 'resnet50':
        resnet50=timm.create_model('resnet50',pretrained=True)
        return resnet50
#model=pre_model('ViT_t_16')
#x=torch.rand([1,3,224,224])
#out=model(x)
#print(out.shape)