import torch
import numpy as np
import display
import cv2


def replicate(x, dim, times):
    x = torch.unsqueeze(x, dim)
    dims = [1] * len(x.shape)
    dims[dim] = times
    x = x.repeat(dims)
    return x

def weight_init(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def showPic(imgs, win=0, name="Real"):
    imgs = [cv2.flip(x.detach().cpu().numpy().transpose(1, 2, 0), 0) for x in imgs]
    half = len(imgs) // 2
    row1 = np.concatenate(imgs[:half], 1)
    row2 = np.concatenate(imgs[half:], 1)
    content = np.concatenate((row1, row2), 0)
    display.image(content, win=win, title=name)