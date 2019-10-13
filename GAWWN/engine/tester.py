import torch
from torch.autograd import Variable
from GAWWN.tools.tools import showPic, to_locs
from GAWWN.tools.config import cfg

def test(netG, dataLoader, device, logger):
    steps = len(dataLoader)
    netG.to(device)
    netG.eval()
    with torch.no_grad():
        for i, (imgs, txts, locs, file_caps, parts_locs, g_locs) in enumerate(dataLoader):
            bs = imgs.shape[0]
            imgs = Variable(imgs).to(device)
            txts = Variable(txts).to(device)
            locs = Variable(locs).to(device)
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

def inference(netG, keyG, dataLoader, device, logger):
    steps = len(dataLoader)
    netG.to(device)
    netG.eval()
    keyG.to(device)
    keyG.eval()
    with torch.no_grad():
        for i, (imgs, txts, locs, file_caps, parts_locs, cond_locs) in enumerate(dataLoader):
            bs = imgs.shape[0]
            imgs = Variable(imgs).to(device)
            txts = Variable(txts).to(device)
            locs = Variable(locs).to(device)
            parts_locs = Variable(parts_locs).to(device)
            cond_locs = Variable(cond_locs).to(device)
            noise = Variable(torch.FloatTensor(bs, cfg.GAN.Z_DIM)).to(device)
            noise.data.normal_(0,1)

            locs_gen = keyG(noise, txts, cond_locs)
            parts_gen = Variable(torch.from_numpy(to_locs(locs_gen.cpu().numpy()))).to(device)

            fake_imgs = netG(txts, parts_gen, noise)

            logger.info("Step: [{iter}/{total}]".format(iter = i + 1, total = steps))
            for i in range(bs):
                logger.info("Filename:{filename}, Cap:{cap}".format(
                    filename = file_caps[0][i], 
                    cap = file_caps[1][i]
                    ))
            
            showPic(imgs[:4].cpu().numpy(), locs[:4].cpu().numpy().sum(1), win=30)
            showPic(fake_imgs[:4].detach().cpu().numpy(), parts_gen[:4].cpu().numpy().sum(1), win=10, name="test")
            input()