import time
import torch

from GAWWN.tools.config import cfg

def train(netG, netD, dataloader, device, optimizerG, optimizerD, criterion, logger):
    logger.info("Start trainning")

    total_step = len(dataloader)
    start_training_time = time.time()
    nepochs = cfg.NEPOCHS

    for epoch in range(nepochs):
        netG.train()
        netD.train()
        for iteration, (imgs, txts, locs, filename, caption) in enumerate(dataloader):
            print((imgs, txts, locs))
            input()
