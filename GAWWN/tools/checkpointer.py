import os
import torch

class Checkpointer():
    def __init__(self, netG, netD, optimizerG=None, optimizerD=None, schedulerG=None, schedulerD=None, logger=None, save_dir="."):
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
        data["netG"] = self.netG.state_dict()
        data["netD"] = self.netD.state_dict()
        if optimizerG is not None:
            data["optimizerG"] = self.optimizerG.state_dict()
        if optimizerD is not None:
            data["optimizerD"] = self.optimizerD.state_dict()
        if schedulerG is not None: 
            data["schedulerG"] = self.schedulerG.state_dict()
        if schedulerD is not None:
            data["schedulerD"] = self.schedulerD.state_dict()

        save_path = os.path.join(self.save_dir, name)
        self.logger.info("Saving checkpointer to {}.pth".format(save_path))
        torch.save(data, save_path)

    def load(self, filepath):
        data = torch.load(filepath, map_location=torch.device("cpu"))
        self.netG.load_state_dict(data["netG"])
        self.netD.load_state_dict(data["netD"])
        if "optimizerG" in data and self.optimizerG is not None:
            optimizerG.load_state_dict(data["optimizerG"])
        if "optimizerD" in data and self.optimizerD is not None:
            optimizerD.load_state_dict(data["optimizerD"])
        if "schedulerG" in data and self.schedulerG is not None:
            schedulerG.load_state_dict(data["schedulerG"])
        if "schedulerD" in data and self.schedulerD is not None:
            schedulerD.load_state_dict(data["schedulerD"])

