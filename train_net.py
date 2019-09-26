import os
import argparse

import torch
from torch.utils.data import DataLoader

from GAWWN.tools.logger import setupLogger
from GAWWN.tools.config import cfg
from GAWWN.model.model_builder import build_models
from GAWWN.dataset.dataset_builder import ImageTextLocDataset
from GAWWN.engine.trainer import train
from GAWWN.tools.checkpointer import Checkpointer


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

    netG, netD, criterion, optimizerG, optimizerD, \
        schedulerG, schedulerD = build_models(device)

    checkpointer = Checkpointer(netG, netD, optimizerG, optimizerD, schedulerG, schedulerD, logger, ouput_dir)
    

    logger.info("dataloader")
    trn_dataset = ImageTextLocDataset(cfg.ROOT_PATH, "all")
    trn_loader = DataLoader(trn_dataset, cfg.BATCH_SIZE, shuffle=True)
    logger.info("data loaded")

    train(netG, netD, trn_loader, device, optimizerG, optimizerD, criterion, schedulerG, schedulerD, logger, checkpointer)


if __name__ == "__main__":
    main()