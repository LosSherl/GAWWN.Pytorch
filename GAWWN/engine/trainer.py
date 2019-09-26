import time
import datetime
import torch
from torch.autograd import Variable

from GAWWN.tools.config import cfg
from GAWWN.tools.tools import showPic

def train(netG, netD, dataloader, device, optimizerG, optimizerD, \
        criterion, schedulerG, schedulerD, logger, checkpointer):
    logger.info("Start trainning")

    total_step = len(dataloader)
    start_training_time = time.time()
    nepochs = cfg.NEPOCHS
    cls_weight = cfg.GAN.CLS_WEIGHT
    fake_score = cfg.GAN.FAKE_SCORE

    for epoch in range(nepochs):
        netG.train()
        netD.train()
        for iteration, (imgs, txts, locs, filename, caption) in enumerate(dataloader):
            bs = imgs.shape[0]
            imgs = Variable(imgs).to(device)
            txts = Variable(txts).to(device)
            locs = Variable(locs).to(device)
            shuf_idx = torch.randperm(bs).to(device)
            txts_shuf = Variable(torch.index_select(txts, 0, shuf_idx)).to(device)

            fake_label = Variable(torch.FloatTensor(bs).fill_(0)).to(device)
            real_label = Variable(torch.FloatTensor(bs).fill_(1)).to(device)
            noise = Variable(torch.FloatTensor(bs, cfg.GAN.Z_DIM)).to(device)

            # Generate fake images
            noise.data.normal_(0, 1)
            fake_imgs = netG(txts, locs, noise)

            # Update D network
            # train with real
            netD.zero_grad()

            output = netD(imgs, txts, locs).view(-1)
            errD_real = criterion(output, real_label)
            
            # train with wrong
            output = netD(imgs, txts_shuf, locs).view(-1)
            errD_wrong = cls_weight * criterion(output, fake_label)

            # train with fake
            output = netD(fake_imgs.detach(), txts, locs).view(-1)
            fake_score = 0.99 * fake_score + 0.01 * output.mean()
            errD_fake = (1 - cls_weight) * criterion(output, fake_label)
            
            errD = errD_real + errD_fake + errD_wrong
            errD.backward()
            optimizerD.step()

            # update G network
            netG.zero_grad()
            output = netD(fake_imgs, txts, locs).view(-1)
            fake_score = 0.99 * fake_score + 0.01 * output.mean()
            errG = criterion(output, real_label)
            errG.backward()
            optimizerG.step()

            if iteration % 1 == 0:
                logger.info(
                    ", ".join(
                        [
                            "Epoch: [{epoch}/{num_epochs}]",
                            "step: [{iter}/{total_step}",
                            "fake_score: {fake_score:.4f}",
                            "lr: {lr:.6f}",
                            "loss G: {loss_g:.4f}",
                            "loss D: {loss_d:.4f}",
                        ]
                    ).format(
                        epoch = epoch + 1, num_epochs = nepochs,
                        iter = iteration + 1, total_step = total_step,
                        fake_score = fake_score, lr = optimizerG.param_groups[0]["lr"],
                        loss_g = errG, loss_d = errD
                    )
                )
                showPic(fake_imgs[:4], win=10)
                showPic(imgs[:4], win=30)
        schedulerG.step()
        schedulerD.step()
        time_spent = time.time() - start_training_time
        logger.info("Epoch:[{}/{}], Time spent {}, Time per epoch {:.4f} s".format(
            epoch + 1, nepochs, str(datetime.timedelta(seconds=time_spent)), time_spent / (epoch + 1)))
        if (epoch + 1) % cfg.CHECKPOINT_PERIOD == 0:
            checkpointer.save("model_{:05d}".format(epoch + 1))








