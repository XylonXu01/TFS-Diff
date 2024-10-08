from io import BytesIO
import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import data.util as Util


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        
        elif datatype == 'Fusion':
            self.First_sr_path = Util.get_paths_from_images(
                '{}/1_sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            
            self.Second_sr_path = Util.get_paths_from_images(
                '{}/2_sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            
            self.Third_sr_path = Util.get_paths_from_images(
                '{}/3_sr_{}_{}'.format(dataroot, l_resolution, r_resolution))

            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            
            if self.need_LR: # 训练时不需要低分辨率图像
                self.First_lr_path = Util.get_paths_from_images(
                    '{}/1_lr_{}'.format(dataroot, l_resolution))
                self.Second_lr_path = Util.get_paths_from_images(
                    '{}/2_lr_{}'.format(dataroot, l_resolution))
                self.Third_lr_path = Util.get_paths_from_images(
                    '{}/3_lr_{}'.format(dataroot, l_resolution))
                
            self.dataset_len = len(self.hr_path)

            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
            
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_Fusion_HR = None
        img_1_LR = None
        img_2_LR = None
        img_3_LR = None
        img_1_SR = None
        img_2_SR = None
        img_3_SR = None
        if self.datatype == 'Fusion':

            img_Fusion_HR = Image.open(self.hr_path[index]).convert("RGB") # 高分辨率
            img_1_SR = Image.open(self.First_sr_path[index]).convert("RGB")
            img_2_SR = Image.open(self.Second_sr_path[index]).convert("RGB")
            img_3_SR = Image.open(self.Third_sr_path[index]).convert("RGB")
            if self.need_LR:
                img_1_LR = Image.open(self.First_lr_path[index]).convert("RGB")
                img_2_LR = Image.open(self.Second_lr_path[index]).convert("RGB")
                img_3_LR = Image.open(self.Third_lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_1_LR, img_2_LR, img_3_LR, img_1_SR,img_2_SR, img_3_SR, img_Fusion_HR] = Util.transform_augment( 
                [img_1_LR, img_2_LR, img_3_LR, img_1_SR,img_2_SR, img_3_SR, img_Fusion_HR], split=self.split, min_max=(-1, 1))
            return {'1_LR': img_1_LR, '2_LR': img_2_LR, '3_LR': img_3_LR, '1_SR': img_1_SR, '2_SR': img_2_SR, '3_SR':img_3_SR, 'HR': img_Fusion_HR, 'Index': index, 
                    '1_2_3_LR': torch.cat([img_1_LR, img_2_LR, img_3_LR], dim=0),
                     '1_2_3_SR': torch.cat([img_1_SR, img_2_SR, img_3_SR], dim=0), }
        else:
            [img_1_SR,img_2_SR, img_3_SR, img_Fusion_HR] = Util.transform_augment( 
                [img_1_SR, img_2_SR, img_3_SR, img_Fusion_HR], split=self.split, min_max=(-1, 1))
            return {'1_SR': img_1_SR, '2_SR': img_2_SR, '3_SR':img_3_SR, 'HR': img_Fusion_HR, 'Index': index, 
                     '1_2_3_SR': torch.cat([img_1_SR, img_2_SR, img_3_SR], dim=0), }
