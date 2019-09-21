import torch
import torch.nn as nn

from GAWWN.tools.config import cfg
from GAWWN.tools.tools import replicate


class Noise_txt(nn.Module):
    def __init__(self):
        super(Noise_txt, self).__init__()
        self.nz = cfg.GAN.Z_DIM 
        self.ngf = cfg.GAN.NGF
        self.nt = cfg.TEXT.TXT_EMBEDDING_DIM

        self.prep_noise = nn.Linear(self.nz, self.ngf * 4)
        self.prep_txt = nn.Linear(self.nt, self.ngf * 4)
        self.ReluAfterCat = nn.ReLU(True)

    def forward(self, txt, z):
        # noise (bs, nz) - text (bs, nt) 

        txt = self.prep_txt(txt)    # (bs, ngf * 4)
        z = self.prep_noise(z)      # (bs, ngf * 4)
        noise_txt = torch.cat((z, txt), 1)      # (bs, ngf * 8)
        noise_txt = self.ReluAfterCat(noise_txt) 

        return noise_txt

class Noise_txt_global(nn.Module):
    def __init__(self, F_noise_txt):
        super(Noise_txt_global, self).__init__()
        self.ngf = cfg.GAN.NGF
        self.num_elt = cfg.KEYPOINT.NUM_ELT

        self.F_noise_txt = F_noise_txt
        self.pre_loc = nn.Sequential(
            nn.Conv2d(self.num_elt, self.ngf, 4, 2, 1),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(self.ngf * 2, self.ngf * 4, 4),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True)
        )
        self.beforeCat = nn.Sequential(
            nn.Linear(self.ngf * 8, self.ngf * 4),
            nn.BatchNorm1d(self.ngf * 4),
            nn.ReLU(True)
        )

        self.upsampling_1 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True)
        )
        self.upsampling_2 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True)
        )
        self.upsampling_3 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True)
        )

    def forward(self, txt, loc, z):
        loc = self.pre_loc(loc)
        loc.view(-1, self.ngf * 4)      # (bs, ngf * 4)

        noise_txt = self.F_noise_txt(txt, z)    # (bs, ngf * 8)
        noise_txt = self.beforeCat(noise_txt)   # (bs, ngf * 4)
        noise_txt_loc = torch.cat((noise_txt, loc), 1)  # (bs, ngf * 8)
        noise_txt_loc = noise_txt_loc.view(-1, self.ngf * 8, 1, 1)
        noise_txt_loc = self.upsampling_1(noise_txt_loc)
        noise_txt_loc = self.upsampling_2(noise_txt_loc)
        noise_txt_loc = self.upsampling_3(noise_txt_loc)    # (bs, ngf, keypoint_dim, keypoint_dim)

        return noise_txt_loc

class Noise_txt_region(nn.Module):
    def __init__(self, F_noise_txt):
        super(Noise_txt_region, self).__init__()
        self.ngf = cfg.GAN.NGF
        self.keypoint_dim = cfg.KEYPOINT.DIM

        self.F_noise_txt = F_noise_txt
        self.before_replicate = nn.Sequential(
            nn.Linear(self.ngf * 8, self.ngf * 4),
            nn.BatchNorm1d(self.ngf * 4),
            nn.ReLU(True)
        )
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True)
        )
 
    def forward(self, txt, loc, z):
        noise_txt = self.F_noise_txt(txt, z)        # (bs, ngf * 8)
        noise_txt = self.before_replicate(noise_txt)    # (bs, ngf * 4)
        noise_txt = replicate(noise_txt, 2, self.keypoint_dim)       # (bs, ngf * 4, keypoint_dim)
        noise_txt = replicate(noise_txt, 3, self.keypoint_dim)       # (bs, ngf * 4, keypoint_dim, keypoint_dim)
        # loc (bs, num_elt, keypoint_dim, keypoint_dim)
        loc = torch.sum(loc, 1)     # (bs, keypoint_dim, keypoint_dim)
        loc = torch.clamp(loc, 0, 1) 
        loc = replicate(loc, 1, self.ngf * 4)  # (bs, ngf * 4, keypoint_dim, keypoint_dim)
        
        noise_txt_loc = noise_txt * loc     # (bs, ngf * 4, keypoint_dim, keypoint_dim)
        noise_txt_loc = self.upsampling(noise_txt_loc)  # (bs, ngf * 4, keypoint_dim, keypoint_dim)

        return noise_txt_loc

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.ngf = cfg.GAN.NGF
        self.num_elt = cfg.KEYPOINT.NUM_ELT
        self.F_noise_txt = Noise_txt()
        self.F_noise_txt_global = Noise_txt_global(self.F_noise_txt)
        self.F_noise_txt_region = Noise_txt_region(self.F_noise_txt)

        self.upsampling_1 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 5 + self.num_elt, self.ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(self.ngf * 4)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf, 1, 1),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, 3, 1, 1),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(self.ngf * 4)
        )

        self.upsampling_2 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 2, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, txt, loc, z):
        ntg = self.F_noise_txt_global(txt, loc, z)    # (bs, ngf, keypoint_dim, keypoint_dim)
        ntr = self.F_noise_txt_region(txt, loc, z)    # (bs, ngf * 4, keypoint_dim, keypoint_dim)
        x = torch.cat((ntg, ntr, loc), 1)           # (bs, ngf * 5 + num_elt, keypoint_dim, keypoint_dim)
        x = x.contiguous()  
        x = self.upsampling_1(x)      # (bs, ngf * 4, keypoint_dim, keypoint_dim)
        x = x + self.conv(x)  # (bs, ngf * 4, keypoint_dim, keypoint_dim)
        x = self.upsampling_2(x) # (bs, 3, imsize, imsize)
        return x
