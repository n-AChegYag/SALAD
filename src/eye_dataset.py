import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class EyeDataset(Dataset):

    def __init__(self, data_path, mode='train'):
        super().__init__()
        self.data_path = data_path
        self.all_images = [os.path.join(data_path, sample) for sample in os.listdir(data_path)]
        self.mode = mode

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        image = cv2.imread(self.all_images[index])[:,:,0]
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        flag = 0
        if self.mode == 'train':
            r = np.random.rand()
            if r >= 0.5:
                flag = 1
                image = self._in_painting(image)
        image = image / 255.0
        image = image[:,:,np.newaxis]
        return image.transpose(2,0,1), flag

    @staticmethod
    def _in_painting(image, patch_num=5, patch_range=(1/8,1/5)):
        W, H = image.shape
        min_size = (W*patch_range[0], H*patch_range[0])
        max_size = (W*patch_range[1], H*patch_range[1])
        for _ in range(patch_num):
            x = np.random.randint(max_size[0], W-max_size[0])
            y = np.random.randint(max_size[1], H-max_size[1])
            patch_w = np.random.randint(min_size[0]/2, max_size[0]/2)
            patch_h = np.random.randint(min_size[1]/2, max_size[1]/2)
            patch_edge= [x-patch_w, x+patch_w, y-patch_h, y+patch_h]
            noise = np.random.randn(patch_edge[1]-patch_edge[0], patch_edge[3]-patch_edge[2]) * 255
            image[patch_edge[0]:patch_edge[1], patch_edge[2]: patch_edge[3]] = noise
        return image


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    eye_dataset = EyeDataset('/home/ps/disk12t/ACY/SALAD/data/eye/A')
    eye_dataloader = DataLoader(eye_dataset, batch_size=4, drop_last=True)
    for image, flag in eye_dataloader:
        print(image.shape, flag)