import os
from skimage import io
import copy
import cv2
import torch
import sys
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as transforms
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, data_dir, mode, logger = None):

        self.data_dir = data_dir
        self.mode = mode
        self.data_path_db = self._data_path_loader()
        self.transform = transforms.Compose([transforms.CenterCrop(self.input_shape), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.logger = logger

    def _data_path_loader(self):
        print('Loading ' + self.mode + ' dataset..')
        if not os.path.isdir(self.data_dir):
            print(f'!!! Cannot find {self.data_dir}... !!!')
            sys.exit()

        print(os.path.join(self.data_dir, self.mode, self.mode + '.pt'))
        if os.path.isfile(os.path.join(self.data_dir, self.mode, self.mode + '.pt')):
            data_path_db = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '.pt'))
        else:
            data_path_db = []
            for (roots, dirs, files) in tqdm(os.walk(os.path.join(self.data_dir, self.mode, 'images'))):
                
                if len(files) != 0:
                    if self.mode == "test":

                        for file in files:
                            img_path = os.path.join(roots, file)
                            data_path_db.append({'img_path': img_path})

                    else:

                        for file in files:
                            img_path = os.path.join(roots, file)
                            data_path_db.append({'img_path': img_path})                     

            torch.save(data_path_db, os.path.join(self.data_dir, self.mode, self.mode + '.pt'))

        return data_path_db

    def __len__(self):
        return len(self.data_path_db)

    def __getitem__(self, index):
        data = copy.deepcopy(self.data_path_db[index])
        cvimg = None

        # 1. load image
        try:
            _ = io.imread(data['img_path'])
            cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        except Exception as e:
            msg = f"Corrupt Image {data['img_path']} at index {index}"
            if self.logger:
                print(msg)
                self.logger.error(msg)
            
            """ if currupt, get data prev or next index  """
            try:
                data = copy.deepcopy(self.data_path_db[index-1])
            except:
                data = copy.deepcopy(self.data_path_db[index+1])

            cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

            # raise IOError("Fail to read %s" % data['img_path'])

        if not isinstance(cvimg, np.ndarray):
            msg = f"Fail to read {data['img_path']}"
            if self.logger:
                print(msg)
                self.logger.error(msg)
            
            """ if img not found, get data prev or next index  """
            try:
                data = copy.deepcopy(self.data_path_db[index-1])
            except:
                data = copy.deepcopy(self.data_path_db[index+1])

            cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        # 2. crop patch from img & generate patch joint ground truth
        img_patch = self.transform(Image.fromarray(cvimg))

        if self.mode == "test":
            raise NotImplementedError
            self.transform = transforms.Compose([transforms.CenterCrop(self.input_shape), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            return img_patch

        elif self.mode == "train":            
            raise NotImplementedError
            self.transform = transforms.Compose([transforms.CenterCrop(self.input_shape), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            return img_patch

        else:
            raise Exception(f"not a valid mode: {self.mode}")

def get_train_loader(data_path, batch_size, num_workers):
    """Returns data set and data loader for training."""
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.3),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(scale=(0.02, 0.12))])
    dataset = Dataset(data_path, transform)
    loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             shuffle=True)
    return dataset, loader


def get_test_loader(csv_path, batch_size, num_workers):
    """Returns data set and data loader for evaluation."""
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.ToTensor()])
    dataset = Dataset(csv_path, transform)
    loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             shuffle=False)
    return dataset, loader


        



