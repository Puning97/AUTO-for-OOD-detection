import numpy as np
import torch.nn.functional as F
def threshold_tool(net,loader,device):
    softmax_score=[]
    net.eval()
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    for batch_idx, (image, label) in enumerate(loader):
        image = image.to(device)
        logits = net(image)
        smax = to_np(F.softmax(logits, dim=1))
        pre_max = np.max(smax, axis=1)

        softmax_score.append(pre_max)
    softmax_score=concat(softmax_score).copy()

    mean=np.array(softmax_score).mean()
    delta=np.array(softmax_score).std()
    print('Mean: {}    Std: {}'.format(mean,delta))
    return mean,delta