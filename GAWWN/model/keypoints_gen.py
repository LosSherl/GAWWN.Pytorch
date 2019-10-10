
import torch
import torch.nn as nn

from GAWWN.tools.tools import replicate
from GAWWN.tools.config import cfg


class keyGen(nn.Module):
    def __init__(self):
        super(keyGen, self).__init__()
        self.nz = cfg.GAN.Z_DIM 
        self.ngf = cfg.GAN.NGF
        self.nt = cfg.TEXT.TXT_EMBEDDING_DIM
        self.num_elt = cfg.KEYPOINT.NUM_ELT
        self.z_enc = nn.Sequential(
            nn.Linear(self.nz, self.ngf * 4),
            nn.ReLU(True)
        )
        self.t_enc = nn.Sequential(
            nn.Linear(self.nt, self.ngf * 4),
            nn.ReLU(True)
        )
        self.loc_enc = nn.Sequential(
            nn.Linear(self.num_elt * 3, self.ngf * 4),
            nn.BatchNorm1d(self.ngf * 4),
            nn.ReLU(True),
            nn.Linear(self.ngf * 4, self.ngf * 2),
            nn.BatchNorm1d(self.ngf * 2),
            nn.ReLU(True)
        )
        
        self.convG = nn.Sequential(
            nn.Linear(self.ngf * 10, self.ngf * 8),
            nn.BatchNorm1d(self.ngf * 8),
            nn.ReLU(True),
            nn.Linear(self.ngf * 8, self.ngf * 4),
            nn.BatchNorm1d(self.ngf * 4),
            nn.ReLU(True),
            nn.Linear(self.ngf * 4, self.ngf * 2),
            nn.BatchNorm1d(self.ngf * 2),
            nn.ReLU(True),
            nn.Linear(self.ngf * 2, self.num_elt * 3)
        )


    def forward(self, noise, txts, locs):
        maskOn = replicate(locs[:,:,-1], 2, 3)
        maskOff = -maskOn + 1
        locs = locs.view(-1, self.num_elt * 3)
        z = self.z_enc(noise)
        t = self.t_enc(txts)
        l = self.loc_enc(locs)
        x = torch.cat((z, t, l), 1)
        x = self.convG(x)
        x = x.view(-1, self.num_elt, 3)
        x = torch.sigmoid(x)
        kg = x * maskOff
        kc = maskOn * locs
        return kg + kc

class keyDis(nn.Module):
    def __init__(self):
        super(keyDis, self).__init__()
        self.ndf = cfg.GAN.NDF
        self.num_elt = cfg.KEYPOINT.NUM_ELT
        self.nt = cfg.TEXT.TXT_EMBEDDING_DIM
        self.loc_enc = nn.Sequential(
            nn.Linear(self.num_elt * 3, self.ndf * 4),
            nn.BatchNorm1d(self.ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.ndf * 4, self.ndf * 2),
            nn.BatchNorm1d(self.ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.t_enc = nn.Sequential(
            nn.Linear(self.nt, self.ndf * 4),
            nn.BatchNorm1d(self.ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.ndf * 4, self.ndf * 2),
            nn.BatchNorm1d(self.ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.judge = nn.Sequential(
            nn.Linear(self.ndf * 4, self.ndf * 2),
            nn.BatchNorm1d(self.ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.ndf * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, locs, txts):
        locs = locs.view(-1, self.num_elt * 3)
        locs = self.loc_enc(locs)
        locs = loc_enc(locs)
        txts = t_enc(txts)
        x = torch.cat((locs, txts), 1)
        return judge(x)