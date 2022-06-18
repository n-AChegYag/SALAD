import torch
import torch.nn as nn
from models import ME, MD, Discriminator, MLP
from losses import SSIM, CTLoss
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import utils
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from eye_dataset import EyeDataset


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    np.random.seed(426)
    torch.manual_seed(426)
    torch.cuda.manual_seed_all(426)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    DATA_PATH = 'data/eye/N'
    LOG_PATH = 'log/220616'
    utils.check_and_create_folder(LOG_PATH)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_dict = {'loss_1':[], 'loss_2':[]}

    batch_size = 64
    lr = 2e-4
    num_workers = 4
    epochs = 200
    alpha = 1
    beta = 10
    gamma = 10
    delta = 10
    eta = 10
    print_freq = 20
    save_freq = 10
    best_loss_1 = np.inf
    best_loss_2 = np.inf

    # dataset
    eye_dataset = EyeDataset(DATA_PATH, 'train')

    # dataloader
    eye_dataloader = DataLoader(eye_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    # losses
    adv_loss_fc = nn.MSELoss()
    str_loss_fc = SSIM()
    fea_loss_fc = nn.L1Loss()
    ct_loss_fc = CTLoss()
    self_loss_fc = nn.L1Loss()

    # models
    m_e = ME(1, 512)
    m_d = MD(512, 1)
    m_d_p = MD(512, 1)
    d_f = MLP(512)
    d_i = Discriminator(1)

    m_e.to(device)
    m_d.to(device)
    m_d_p.to(device)
    d_f.to(device)
    d_i.to(device)
    
    part_1 = [m_e, m_d, m_d_p]
    part_2 = [d_f, d_i]

    optim_params_1 = [{'params': net.parameters()} for net in part_1]
    optimizer_1 = optim.Adam(optim_params_1, lr)

    optim_params_2 = [{'params': net.parameters()} for net in part_2]
    optimizer_2 = optim.Adam(optim_params_2, lr)

    loss_1_am = utils.AverageMeter()
    loss_2_am = utils.AverageMeter()

    plt.figure(figsize=(10, 6))
    for epoch in range(1, epochs+1):
        loss_1_am.reset()
        loss_2_am.reset()
        for idx, (image, flag) in enumerate(eye_dataloader):
            image = ((image - 0.5) * 2).float()
            image = image.to(device)
            # train G
            Z = m_e(image)
            z_dict, augmented_image_num = utils.choose_augmented_samples(Z, flag, batch_size)
            z_p = z_dict['ori'].to(device)
            z_s_p = z_dict['aug'].to(device)
            x_dict, augmented_image_num = utils.choose_augmented_samples(image, flag, batch_size)
            x = x_dict['ori'].to(device)
            x_s = x_dict['aug'].to(device)
            x_h = m_d(z_p)
            x_s_p = m_d_p(z_s_p)
            z = torch.randn((batch_size, 512, 1, 1)).to(device)
            x_p = m_d(z)
            z_h = m_e(x_p)
            rf_f = d_f(z_p.squeeze())
            rf_i = d_i(x_p)
            adv_loss = 0.5*adv_loss_fc(rf_f, torch.ones_like(rf_f, requires_grad=False)) + 0.5*adv_loss_fc(rf_i, torch.zeros_like(rf_i, requires_grad=False))
            str_loss = str_loss_fc(x_h, x)
            fea_loss = fea_loss_fc(z_h, z)
            ct_loss = ct_loss_fc(z_p)
            self_loss = self_loss_fc(x_s_p, x_s)
            loss_1 = alpha*adv_loss + beta*str_loss + gamma*fea_loss + delta*ct_loss + eta*self_loss
            loss_1_am.update(loss_1.item(), batch_size)
            optimizer_1.zero_grad()   
            loss_1.backward()
            optimizer_1.step()
            if idx % print_freq == 0:
                print(
                    'G: [{}] | [{} / {}]'.format(epoch, idx, len(eye_dataloader)),
                    'loss: {:.4f} ({:.4f})'.format(loss_1_am.val, loss_1_am.avg),
                    'lr: {:.6f}'.format(optimizer_1.param_groups[0]['lr']),
                    sep='\n'
                )
            # train D
            z_p = m_e(x).detach()
            x_h = m_d(z_p).detach()
            z = torch.randn((batch_size, 512, 1, 1)).to(device)
            x_p = m_d(z).detach()
            z_h = m_e(x_p).detach()
            rf_f_real = d_f(z_p.squeeze())
            rf_f_fake = d_f(z.squeeze())
            rf_i_real = d_i(x)
            rf_i_fake = d_i(x_p)
            loss_f_real = adv_loss_fc(rf_f_real, torch.ones_like(rf_f_real, requires_grad=False))
            loss_f_fake = adv_loss_fc(rf_f_fake, torch.zeros_like(rf_f_fake, requires_grad=False))
            loss_i_real = adv_loss_fc(rf_i_real, torch.ones_like(rf_i_real, requires_grad=False))
            loss_i_fake = adv_loss_fc(rf_i_fake, torch.zeros_like(rf_i_fake, requires_grad=False))
            loss_2 = loss_f_real + loss_f_fake + loss_i_real + loss_i_fake
            loss_2_am.update(loss_2.item(), batch_size - augmented_image_num)
            optimizer_2.zero_grad()   
            loss_2.backward()
            optimizer_2.step()
            if idx % print_freq == 0:
                print(
                    'D: [{}] | [{} / {}]'.format(epoch, idx, len(eye_dataloader)),
                    'loss: {:.4f} ({:.4f})'.format(loss_2_am.val, loss_2_am.avg),
                    'lr: {:.6f}'.format(optimizer_2.param_groups[0]['lr']),
                    sep='\n'
                )

        if epoch % save_freq == 0:
            save_image(x_h / 2 + 0.5, os.path.join(LOG_PATH, 'x_h_{}.jpg'.format(epoch)))
            save_image(x_p / 2 + 0.5, os.path.join(LOG_PATH, 'x_p_{}.jpg'.format(epoch)))
            save_image(x_s / 2 + 0.5, os.path.join(LOG_PATH, 'x_s_{}.jpg'.format(epoch)))
            save_image(x_s_p / 2 + 0.5, os.path.join(LOG_PATH, 'x_s_p_{}.jpg'.format(epoch)))

        # save checkpoint
        loss_dict['loss_1'].append(loss_1_am.avg)
        loss_dict['loss_2'].append(loss_2_am.avg)
        utils.plot_loss_curve(loss_dict['loss_1'], loss_dict['loss_2'], LOG_PATH)
        np.save(os.path.join(LOG_PATH, 'loss.npy'), loss_dict) 

        if loss_1_am.avg < best_loss_1:
            best_loss_1 = loss_1_am.avg
            is_best = True
        else:
            is_best = False
        state = {
            'm_e': m_e.state_dict(),
            'm_d': m_d.state_dict(),
            'm_d_p': m_d_p.state_dict(),
            'd_f': d_f.state_dict(),
            'd_i': d_i.state_dict(),
        }
        utils.save_checkpoint(state=state, is_best=is_best, save_path=LOG_PATH)









    
