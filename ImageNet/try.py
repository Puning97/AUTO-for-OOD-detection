import torch,timm
from model_pre.resnet import resnet50
model=resnet50(pretrained=True)
x= torch.rand(2,3,224,224)
y,list1=model.feature_list(x)
print(len(list1))
print(list1[0].shape)
model2=timm.create_model('vit_small_patch16_224',pretrained=True)
y2=model2.forward_features(x)
print(y2.shape)