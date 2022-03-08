import torch
import numpy as np
from torch.utils.data import Dataset

class UNet3DDataset(Dataset):
    def __init__(self, scans_segs, size= None):
        self.img_dir = []
        for s in scans_segs:
            # if each image is different sized, we need to get w, h here
            _, width, height = s[0].shape
            # add channel dimension to scans
            x = s[0].reshape((-1, width, height, 1))
            y = s[1]

            diff = 32 - x.shape[0]
            if diff > 0:
                x = np.concatenate((x, np.zeros((diff, width, height, 1))), axis =0) 
                y = np.concatenate((y, np.zeros((diff, width, height, 3))), axis =0) 

            x = x.transpose()
            y = y.transpose()
            self.img_dir.append((torch.from_numpy(x).float(), torch.from_numpy(y).float()))


    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        return self.img_dir[idx][0], self.img_dir[idx][1]


