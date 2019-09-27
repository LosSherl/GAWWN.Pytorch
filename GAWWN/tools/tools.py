import torch
import torch.nn as nn
import numpy as np
import display
import cv2


def replicate(x, dim, times):
    x = torch.unsqueeze(x, dim)
    dims = [1] * len(x.shape)
    dims[dim] = times
    x = x.repeat(dims)
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def showPic(imgs, locs, win=0, name="Real"):
    imgs = [cv2.flip(x.transpose(1, 2, 0), 0) for x in imgs]
    for i in range(4):
        locs[i] = locs[i].sum(1)
        for y in range(16):
            for x in range(16):
                if locs[i][y][x] > 0:
                    cv2.rectangle(imgs[i], (x * 8, y * 8), (8,8), (0, 0, 255), 1)
    half = len(imgs) // 2
    row1 = np.concatenate(imgs[:half], 1)
    row2 = np.concatenate(imgs[half:], 1)
    content = np.concatenate((row1, row2), 0)
    display.image(content, win=win, title=name)