import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import settings as settings
from natsort import natsorted


class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.root_dir = os.path.join(settings.root_dir, name)
        self.mat_files = os.listdir(self.root_dir)
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        B = np.transpose(img_pair[:, w:], (2, 0, 1))
        O = np.transpose(img_pair[:, :w], (2, 0, 1))
        sample = {'O': O, 'B': B}

        return sample


class ShowDataset(Dataset):
    def __init__(self,name):
        super().__init__()
        self.root_dir = settings.real_dir
        self.mat_files_rain= natsorted(os.listdir(self.root_dir))
        self.file_num = len(self.mat_files_rain)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files_rain[idx % self.file_num]
        img_file_dir = os.path.join(self.root_dir, file_name)
        img_file = cv2.imread(img_file_dir).astype(np.float32) / 255
        O = np.transpose(img_file, (2, 0, 1))
        sample = {'O': O,  'file_name': file_name}
        return sample


