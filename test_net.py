import os
import argparse
import torch
from torch.utils.data import DataLoader

from GAWWN.dataset.dataset_builder import ImageTextLocDataset
from GAWWN.model.discriminator import Dis
from GAWWN.model.generator import Gen
from GAWWN.tools.config import cfg
from GAWWN.tools.logger import setupLogger
from GAWWN.tools.checkpointer import Checkpointer
from GAWWN.engine.tester import test

def main():
    parser = argparse.ArgumentParser(description="GAWWN")
    parser.add_argument(
        "-m",
        dest="model_path",
        default="output/GAWWN/model_01000.pth"
    )
    parser.add_argument(
        "-p",
        dest="root_path"
    )
    parser.add_argument(
        "-bs",
        dest = "batch_size",
        default = 4,
        type = int
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
    netD = Dis()

    checkpointer = Checkpointer(netG, netD)
    checkpointer.load(cfg.MODEL_PATH)

    dataset = ImageTextLocDataset(cfg.ROOT_PATH, "all")
    dataLoader = DataLoader(dataset, cfg.BATCH_SIZE, shuffle=True, num_workers=4)

    test(netG, dataLoader, device, logger)
    

if __name__ == "__main__":
    main()