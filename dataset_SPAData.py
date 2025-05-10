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
        self.root_dir_rain = os.path.join(self.root_dir, "rain")
        self.root_dir_label = os.path.join(self.root_dir, "norain")
        self.mat_files_rain = sorted(os.listdir(self.root_dir_rain))
        self.file_num = len(self.mat_files_rain)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name_rain = self.mat_files_rain[idx % self.file_num]
        file_name_label = file_name_rain.split('.')[0] + 'gt.png'
        img_file_rain = os.path.join(self.root_dir_rain, file_name_rain)
        img_file_label = os.path.join(self.root_dir_label, file_name_label)
        img_rain = cv2.imread(img_file_rain).astype(np.float32) / 255
        img_label = cv2.imread(img_file_label).astype(np.float32) / 255
        B = np.transpose(img_label, (2, 0, 1))
        O = np.transpose(img_rain, (2, 0, 1))
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


