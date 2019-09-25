import time
import datetime
import torch
from torch.autograd import Variable


from GAWWN.tools.config import cfg

def train(netG, netD, dataloader, device, optimizerG, optimizerD, \
        criterion, schedulerG, schedulerD, logger, checkpointer):
    logger.info("Start trainning")

    total_step = len(dataloader)
    start_training_time = time.time()
    nepochs = cfg.NEPOCHS
    cls_weight = cfg.GAN.CLS_WEIGHT
    batch_size = cfg.BATCH_SIZE
    fake_score = cfg.GAN.FAKE_SCORE

    for epoch in range(nepochs):
        netG.train()
        netD.train()

        label = Variable(torch.FloatTensor(batch_size).fill_(0))
        noise = Variable(torch.FloatTensor(batch_size, cfg.GAN.Z_DIM))

        for iteration, (imgs, txts, locs, filename, caption) in enumerate(dataloader):
            imgs = Variable(imgs).to(device)
            txts = Variable(txts).to(device)
            locs = Variable(locs).to(device)
            shuf_idx = torch.randperm(batch_size)
            txts_shuf = Variable(torch.index_select(txts, 0, shuf_idx)).to(device)
            
            # Generate fake images
            noise.data.normal_(0, 1)
            fake_imgs = netG(txts, locs, noise)

            # Update D network
            # train with real
            label.data.fill_(1)
            netD.zero_grad()

            output = netD(imgs, locs, txts)
            errD_real = criterion(output, label)
            
            # train with wrong
            label.data.fill_(0)
            output = netD(imgs, locs, txts_shuf)
            errD_wrong = cls_weight * criterion(output, label)

            # train with fake
            # label.data.fill_(0)
            output = netD(fake_imgs, locs, txts)
            fake_score = 0.99 * fake_score + 0.01 * output.mean()
            errD_fake = (1 - cls_weight) * criterion(output, label)
            
            errD = errD_real + errD_fake + errD_wrong
            errD.backward()
            optimizerD.step()

            # update G network
            label.data.fill_(1)
            netG.zero_grad()
            fake_score = 0.99 * fake_score + 0.01 * output.mean()
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            if iteration % 10 == 0:
                logger.info(", ".join(
                    [
                        "Epoch: [{epoch}/{num_epochs}]",
                        "Step: [{iter}/{total_step}",
                        "Fake_score: {fake_score:.4f}",
                        "lr: {lr:.6f}",
                        "loss G: {loss_g:.4f}%",
                        "loss D: {loss_d:.4f}",
                    ]
                )).format(
                    epoch = epoch + 1, num_epochs = nepochs,
                    iter = iteration + 1, total_step = total_step,
                    fake_score = fake_score, lr = optimizerG.param_groups[0]["lr"],
                    loss_g = errG, loss_d = errD
                )
                # display.image()
        schedulerG.step()
        schedulerD.step()
        time_spent = time.time() - start_training_time
        logger.info("Epoch:[{}/{}], Time spent {}, Time per epoch {:.4f} s".format(
            epoch + 1, nepochs, str(datetime.timedelta(seconds=time_spent)), time_spent / (epoch + 1)))
        if (epoch + 1) % 100 == 0:
            checkpointer.save("model_{:05d}".format(epoch + 1))








