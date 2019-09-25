import os
import torch

class Checkpointer():
    def __init__(self, netG, netD, optimizerG, optimizerD, schedulerG, schedulerD, logger, save_dir):
        self.netG = netG
        self.netD = netD
        self.optimizerG = optimizerG
        self.optimizerD = optimizerD
        self.schedulerG = schedulerG
        self.schedulerD = schedulerD
        self.logger = logger
        self.save_dir = save_dir
    
    def save(self, name):
        data = dict()
        data["netG"] = self.netG
        data["netD"] = self.netD
        data["optimizerG"] = self.optimizerG
        data["optimizerD"] = self.optimizerD
        data["schedulerG"] = self.schedulerG
        data["schedulerD"] = self.schedulerD

        save_path = os.path.join(self.save_dir, name)
        self.logger.info("Saving checkpointer to {}".format(save_path))
        torch.save(data, save_path)

