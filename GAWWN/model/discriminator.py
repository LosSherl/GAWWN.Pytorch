import torch
import torch.nn as nn

from GAWWN.tools.config import cfg
from GAWWN.tools.tools import replicate

class keyMulD(nn.Module):
    def __init__(self, F_imgGlobal, F_prep_txtD):
        super(keyMulD, self).__init__()
        self.ndf = cfg.GAN.NDF
        self.nt_d = cfg.TEXT.TXT_FEATURE_DIM
        self.F_imgGlobal = F_imgGlobal
        self.keypoint_dim = cfg.KEYPOINT.DIM
        
        self.F_prep_txtD = F_prep_txtD
        self.conv = nn.Sequential(
            nn.Conv2d(self.nt_d + self.ndf * 2, self.ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, img, loc, txt):
        imgGlobal = self.F_imgGlobal(img)   # (bs, ndf * 2, 16, 16)
        prep_txt_d = self.F_prep_txtD(txt)    # (bs, nt_d)
        prep_txt_d = replicate(prep_txt_d, 2, self.keypoint_dim)    # (bs, nt_d, 16)
        prep_txt_d = replicate(prep_txt_d, 3, self.keypoint_dim)    # (bs, nt_d, 16, 16)
        imgTextGlobal = torch.cat((imgGlobal, prep_txt_d), 1)       # (bs, nt_d + ndf * 2, 16, 16)
        imgTextGlobal = self.conv(imgTextGlobal)    # (bs, ndf * 2, 16, 16)
        
        # loc (bs, num_elt, keypoint_dim, keypoint_dim)
        loc = torch.sum(loc, 1)     # (bs, keypoint_dim, keypoint_dim)
        loc = torch.clamp(loc, 0, 1) 
        loc = replicate(loc, 1, self.ndf * 2)

        x = imgTextGlobal * loc
        return x 


class regionD(nn.Module):
    def __init__(self, F_imgGlobal, F_prep_txtD):
        super(regionD, self).__init__()
        self.ndf = cfg.GAN.NDF
        self.num_elt = cfg.KEYPOINT.NUM_ELT

        self.F_KeyMulD = keyMulD(F_imgGlobal, F_prep_txtD)
        self.conv = nn.Sequential(
            nn.Conv2d(self.ndf * 2 + self.num_elt, self.ndf * 2, 1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf * 2, self.ndf, 2)
        )
        self.LReLU = nn.LeakyReLU(0.2, True)

    def forward(self, img, loc, txt):
        keyMul = self.F_KeyMulD(img, loc, txt)
        x = torch.cat((keyMul, loc), 1)        # (bs, ngf * 2 + num_elt, 16, 16)
        x = x.contiguous()
        x = self.conv(x)
        x = x.mean(3)
        x = x.mean(2)
        x = self.LReLU(x)
        return x


class globalD(nn.Module):
    def __init__(self, F_imgGlobal, F_prep_txtD):
        super(globalD, self).__init__()
        self.ndf = cfg.GAN.NDF
        self.nt_d = cfg.TEXT.TXT_FEATURE_DIM
        self.F_imgGlobal = F_imgGlobal
        self.F_prep_txtD = F_prep_txtD

        self.convGlobal = nn.Sequential(
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.ndf * 8 + self.nt_d, self.ndf * 4, 1),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf * 4, self.ndf, 4),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, img, txt):
        img = self.F_imgGlobal(img)     # (bs, ndf * 2, 16, 16)
        img = self.convGlobal(img)      # (bs, ndf * 8, 4, 4)

        txtGlobal = self.F_prep_txtD(txt)      # (bs, nt)
        txtGlobal = replicate(txtGlobal, 2, 4)      # (bs, nt_d, 4)
        txtGlobal = replicate(txtGlobal, 3, 4)      # (bs, nt_d, 4, 4)

        imgTxtGlobal = torch.cat((img, txtGlobal), 1)     # (bs, nt_d + ndf * 8, 4 ,4)
        imgTxtGlobal = imgTxtGlobal.contiguous()
        imgTxtGlobal = self.conv(imgTxtGlobal)  # (bs, ndf, 1, 1)
        imgTxtGlobal = imgTxtGlobal.view(-1, self.ndf)
        
        return imgTxtGlobal

class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.ndf = cfg.GAN.NDF
        self.nt = cfg.TEXT.TXT_EMBEDDING_DIM
        self.nt_d = cfg.TEXT.TXT_FEATURE_DIM

        self.prep_txtD = nn.Sequential(
            nn.Linear(self.nt, self.nt_d),
            nn.LeakyReLU(0.2, True)
        )
        self.imgGlobalD = nn.Sequential( 
            nn.Conv2d(3, self.ndf, 4, 2, 1), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf, self.ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.ndf * 2, self.ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, True)
        )

        self.F_regionD = regionD(self.imgGlobalD, self.prep_txtD) 
        self.F_globalD = globalD(self.imgGlobalD, self.prep_txtD)
        self.judge = nn.Sequential(
            nn.Linear(self.ndf * 2, self.ndf),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, True),
            nn.Linear(self.ndf, 1),
            nn.Sigmoid()
        )

    def forward(self, img, loc, txt):
        region_d = self.F_regionD(img, loc, txt)
        global_d = self.F_globalD(img, txt)
        x = torch.cat((region_d, global_d), 1)
        x = self.judge(x)
        return x