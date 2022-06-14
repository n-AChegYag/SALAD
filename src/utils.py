import os
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np

def check_and_create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def plot_loss_curve(loss_1, loss_2, save_path):
    '''
    precondition: plt.figure(figsize=(10, 6))
    '''
    y1 = np.array(loss_1)
    y2 = np.array(loss_2)
    x1 = range(0,len(y1))
    x2 = range(0,len(y2))
    plt.plot(x1, y1, '', label='loss_1')
    plt.plot(x2, y2, '', label='loss_2')
    plt.title('loss')
    plt.legend(loc='upper right')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(save_path, 'loss.png'))
    plt.clf()

def save_checkpoint(state, is_best, save_path):
    torch.save(state['m_e'], os.path.join(save_path,'m_e.pkl'))
    torch.save(state['m_d'],os.path.join(save_path,'m_d.pkl'))
    torch.save(state['m_d_p'], os.path.join(save_path,'m_d_p.pkl'))
    torch.save(state['d_f'],os.path.join(save_path,'d_f.pkl'))
    torch.save(state['d_i'], os.path.join(save_path,'d_i.pkl'))
    torch.save(state, os.path.join(save_path, 'model.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(save_path, 'model.pth.tar'), os.path.join(save_path,'model_best.pth.tar'))
        shutil.copyfile(os.path.join(save_path, 'm_e.pkl'), os.path.join(save_path,'m_e_best.pkl'))
        shutil.copyfile(os.path.join(save_path, 'm_d.pkl'), os.path.join(save_path,'m_d_best.pkl'))
        shutil.copyfile(os.path.join(save_path, 'm_d_p.pkl'), os.path.join(save_path,'m_d_p_best.pkl'))
        shutil.copyfile(os.path.join(save_path, 'd_f.pkl'), os.path.join(save_path,'d_f_best.pkl'))
        shutil.copyfile(os.path.join(save_path, 'd_i.pkl'), os.path.join(save_path,'d_i_best.pkl'))

class AverageMeter(object):
    '''
    Computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def choose_augmented_samples(image, flag, batch_size):
    data_dict = {}
    augmented_image_num =flag.sum().item()
    if augmented_image_num == batch_size:
        data_dict['aug'] = image
        data_dict['ori'] = None
        return data_dict, augmented_image_num
    elif augmented_image_num == 0:
        data_dict['aug'] = None
        data_dict['ori'] = image
        return data_dict, augmented_image_num
    else:
        aug_data = torch.zeros((augmented_image_num, image.size()[1], image.size()[2], image.size()[3]))
        ori_data = torch.zeros((batch_size-augmented_image_num, image.size()[1], image.size()[2], image.size()[3]))
        try:
            aug_index = torch.where(flag==1)[0].cpu().numpy().tolist()
            ori_index = torch.where(flag==0)[0].cpu().numpy().tolist()
        except:
            aug_index = torch.where(flag==1)[0].numpy().tolist()
            ori_index = torch.where(flag==0)[0].numpy().tolist()
        for idx, i in enumerate(aug_index):
            aug_data[idx,:,:,:] = image[i,:,:,:]
        for idx, i in enumerate(ori_index):
            ori_data[idx,:,:,:] = image[i,:,:,:]
        data_dict['aug'] = aug_data
        data_dict['ori'] = ori_data
        return data_dict, augmented_image_num