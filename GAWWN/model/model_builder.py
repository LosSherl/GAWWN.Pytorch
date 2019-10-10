import torch
from GAWWN.model.generator import Gen
from GAWWN.model.discriminator import Dis
from GAWWN.model.keypoints_gen import keyGen, keyDis
from GAWWN.tools.tools import weights_init
from GAWWN.tools.config import cfg

def build_models(device):
    if cfg.MODE == "img":
        netG = Gen()
        netD = Dis()
    else:
        netG = keyGen()
        netD = keyDis()
    netG.apply(weights_init)
    netD.apply(weights_init)
    netG.to(device)
    netD.to(device)
    criterion = torch.nn.BCELoss().to(device)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=cfg.LR, betas=cfg.BETAS)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=cfg.LR, betas=cfg.BETAS)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=cfg.DECAY_PERIOD, gamma=cfg.LR_DECAY)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=cfg.DECAY_PERIOD, gamma=cfg.LR_DECAY)
    
    return netG, netD, criterion, optimizerG, optimizerD, schedulerG, schedulerD