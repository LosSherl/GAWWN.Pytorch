import torch
import torch.nn as nn
import numpy as np
import display
import cv2
from GAWWN.tools.config import cfg


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

def to_locs(parts):
    imsize = cfg.IMAGE.FINESIZE
    keypoint_dim =  cfg.KEYPOINT.DIM
    locs = np.zeros((parts.shape[0], cfg.KEYPOINT.NUM_ELT, keypoint_dim, keypoint_dim), dtype="float32")
    for i in range(parts.shape[0]):
        for j in range(cfg.KEYPOINT.NUM_ELT):
            if parts[i][j][2] > 0.5:
                x = min(keypoint_dim - 1, round(parts[i][j][0] * keypoint_dim))
                y = min(keypoint_dim - 1, round(parts[i][j][1] * keypoint_dim))
                locs[i][j][int(y)][int(x)] = 1
    return locs

def showPic(imgs, locs, win=0, name="Real"):
    imgs = [(x + 0.5) / 2 * 255 for x in imgs]
    imgs = [cv2.flip(x.transpose(1, 2, 0), 0) for x in imgs]
    # imgs = [x.transpose(1, 2, 0) for x in imgs]
    for i in range(4):
        for y in range(16):
            for x in range(16):
                if locs[i][y][x] > 0.3:
                    cv2.rectangle(imgs[i], (x * 8, 127 - y * 8), (x * 8 + 8, 119 - y * 8), (0, 0, 255), 1)
                    # cv2.rectangle(imgs[i], (x * 8,  y * 8), (x * 8 + 8, y * 8 + 8), (0, 0, 255), 1)
    half = len(imgs) // 2
    row1 = np.concatenate(imgs[:half], 1)
    row2 = np.concatenate(imgs[half:], 1)
    content = np.concatenate((row1, row2), 0)
    try:
        display.image(content, win=win, title=name)
    except Exception as e:
        print(e)
    