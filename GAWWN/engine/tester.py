import torch
from torch.autograd import Variable
from GAWWN.tools.tools import showPic
from GAWWN.tools.config import cfg

def test(netG, dataLoader, device, logger):
    steps = len(dataLoader)
    netG.to(device)
    netG.eval()
    with torch.no_grad():
        for i, (imgs, txts, locs, file_caps) in enumerate(dataLoader):
            bs = imgs.shape[0]
            imgs = Variable(imgs).to(device)
            txts = Variable(txts).to(device)
            locs = Variable(locs).to(device)
            # locs[0][1].fill_(0.0)
            # locs[0][13].fill_(0.0)
            # locs[0][13][13][13] = 1
            # locs[0][1][3][3] = 1
            noise = Variable(torch.FloatTensor(bs, cfg.GAN.Z_DIM)).to(device)
            noise.data.normal_(0,1)
            fake_imgs = netG(txts, locs, noise)

            logger.info("Step: [{iter}/{total}]".format(iter = i + 1, total = steps))
            for i in range(bs):
                logger.info("Filename:{filename}, Cap:{cap}".format(
                    filename = file_caps[0][i], 
                    cap = file_caps[1][i]
                    ))
            
            showPic(imgs[:4].cpu().numpy(), locs[:4].cpu().numpy().sum(1), win=30)
            showPic(fake_imgs[:4].detach().cpu().numpy(), locs[:4].cpu().numpy().sum(1), win=10, name="test")
            input()