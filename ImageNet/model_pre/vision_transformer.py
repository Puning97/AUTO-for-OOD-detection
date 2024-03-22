import timm
import torch
import torch.nn as nn

ViT_b_16_para=timm.create_model('vit_base_patch16_224',pretrained=True)
torch.save(ViT_b_16_para.state_dict(),'/data/8_imagenet_online/model_pre/ViT.pth')

class ViT(nn.Module):
    def __init__(self, num_classes=1000):
        super(ViT, self).__init__()
        self.feature_extractor= timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.head = nn.Linear(768, num_classes)

    def forward(self,x):
        x=self.feature_extractor(x)
        print(x.shape)
        x=x.clip(max=1.0)
        x=self.head(x)
        return x

pretrained_dict = torch.load('/data/8_imagenet_online/model_pre/ViT.pth')

def vit_our():
    model=ViT(num_classes=1000)
    model_dict=model.state_dict()
    keys = []
    for k, v in pretrained_dict.items():
        keys.append(k)
    i = 0
    for k, v in model_dict.items():
        if v.size() == pretrained_dict[keys[i]].size():
            model_dict[k] = pretrained_dict[keys[i]]
            # print(model_dict[k])
            i = i + 1
    model.load_state_dict(model_dict)
    return model

#for k,v in model_our.named_parameters():
    #if 'feature_extractor' in k:
        #k=k.replace('feature_extractor.','')
#model_our.load_state_dict(m.state_dict())