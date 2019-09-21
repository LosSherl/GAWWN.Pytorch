import os
import datetime
import time
import logging
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from GAWWN.tools.logger import setupLogger
from GAWWN.tools.config import cfg
from GAWWN.tools.tools import weight_init
from GAWWN.model.generator import Gen
from GAWWN.model.discriminator import Dis
from GAWWN.dataset.dataset_builder import ImageTextLocDataset
from GAWWN.engine.trainer import train

def main():
    parser = argparse.ArgumentParser(description="GAWWN")
    parser.add_argument(
        "-name",
        dest="name"
    )
    parser.add_argument(
        "-p", 
        dest = "root_path"
    )
    parser.add_argument(
        "-bs",
        dest = "batch_size",
        type = int
    )
    parser.add_argument(
        "-n",
        dest = "nepochs",
        type = int
    )
    parser.add_argument(
        "-cp",
        dest = "checkpoint_period",
        type = int
    )
    parser.add_argument(
        "-lr",
        dest = "lr",
        type = float
    )

    args = parser.parse_args()
    args = vars(args)
    for key in args:
        if args[key] is not None:
            cfg[key.upper()] = args[key]
    
    ouput_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME)
    logger = setupLogger(cfg.NAME, ouput_dir)
    logger.info(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = Gen()
    netG.apply(weight_init)
    netD = Dis()
    netD.apply(weight_init)
    netG.to(device)
    netD.to(device)

    logger.info("dataloader")
    trn_dataset = ImageTextLocDataset(cfg.ROOT_PATH, "train")
    trn_loader = DataLoader(trn_dataset, cfg.BATCH_SIZE, shuffle=True)
    logger.info("data loaded")

    criterion = nn.BCELoss().to(device)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=cfg.LR, betas=cfg.BETAS)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=cfg.LR, betas=cfg.BETAS)

    train(netG, netD, trn_loader, device, optimizerG, optimizerD, criterion, logger)

    
if __name__ == "__main__":
    main()