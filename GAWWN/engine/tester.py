import torch
from torch.autograd import Variable
from GAWWN.tools.tools import show_pic
from GAWWN.tools.config import cfg

def test(netG, dataLoader, device, logger):
    steps = len(dataLoader)
    netG.val()
    for i, (imgs, txts, locs, file_caps) in enumerate(dataLoader):
        bs = imgs.shape[0]
        imgs = Variable(imgs).to(device)
        txts = Variable(txts).to(device)
        locs = Variable(locs).to(device)
        noise = Variable(torch.FloatTensor(bs, cfg.GAN.Z_DIM)).to(device)
        noise.data.normal_(0,1)
        fake_img = netG(txts, locs, noise)

        logger.info("Step: [{iter}/{total}]".format(iter = i + 1, total = steps))
        for item in file_caps:
            logger.info("Filename:{filnename}, Cap:{cap}".format(Filename = item[0], item[1]))
         
        showPic(imgs[:4].cpu().numpy(), locs[:4].cpu().numpy().sum(1), win=30)
        showPic(fake_imgs[:4].detach().cpu().numpy(), locs[:4].cpu().numpy().sum(1), win=10, name="test")