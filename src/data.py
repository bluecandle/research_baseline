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
    def __init__(self, data_dir, mode, transform=None, logger=None):

        self.data_dir = data_dir
        self.mode = mode
        self.data_path_db = self._data_path_loader()
        self.base_transform = transforms.Compose(
            [
                transforms.CenterCrop(self.input_shape),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.transform = transform
        self.logger = logger

    def _data_path_loader(self):
        print("Loading " + self.mode + " dataset..")
        print(os.path.join(self.data_dir, self.mode, self.mode + ".pt"))

        if not os.path.isdir(self.data_dir):
            print(f"!!! Cannot find {self.data_dir}... !!!")
            sys.exit()

        if os.path.isfile(os.path.join(self.data_dir, self.mode, self.mode + ".pt")):
            data_path_db = torch.load(
                os.path.join(self.data_dir, self.mode, self.mode + ".pt")
            )

        else:
            data_path_db = []
            img_dir = os.path.join(self.data_dir, self.mode, "images")
            for (roots, dirs, files) in tqdm(os.walk(img_dir)):

                if len(files) != 0:
                    if self.mode == "test":
                        # test

                        for file in files:
                            img_path = os.path.join(roots, file)
                            data_path_db.append({"img_path": img_path})

                    else:
                        # train
                        # annot

                        for file in files:
                            img_path = os.path.join(roots, file)

                            raise NotImplementedError
                            annot_path = None
                            data_path_db.append({"img_path": img_path})
                            data_path_db.append({"annot": annot_path})

            torch.save(
                data_path_db, os.path.join(self.data_dir, self.mode, self.mode + ".pt")
            )

        return data_path_db

    def __len__(self):
        return len(self.data_path_db)

    def __getitem__(self, index):
        data = copy.deepcopy(self.data_path_db[index])
        annot = data["annot"]
        cvimg = None

        # 1. load image
        try:
            _ = io.imread(data["img_path"])
            # cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            cvimg = cv2.imread(data["img_path"])

        except Exception as e:
            msg = f"Corrupt Image {data['img_path']} at index {index}"
            if self.logger:
                print(msg)
                self.logger.error(msg)

            """ if currupt, get data prev or next index  """
            try:
                data = copy.deepcopy(self.data_path_db[index - 1])
            except:
                data = copy.deepcopy(self.data_path_db[index + 1])

            # cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            cvimg = cv2.imread(data["img_path"])

            # raise IOError("Fail to read %s" % data['img_path'])

        if not isinstance(cvimg, np.ndarray):
            msg = f"Fail to read {data['img_path']}"
            if self.logger:
                print(msg)
                self.logger.error(msg)

            """ if img not found, get data prev or next index  """
            try:
                data = copy.deepcopy(self.data_path_db[index - 1])
            except:
                data = copy.deepcopy(self.data_path_db[index + 1])

            # cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            cvimg = cv2.imread(data["img_path"])

        if self.transform:
            img = self.transform(Image.fromarray(cvimg))

        else:
            img = self.base_transform(Image.fromarray(cvimg))

            # if self.mode == "train":
            # print("transform not given, base transform used")

        if self.mode == "test":

            return img

        elif self.mode == "train":

            return img, annot

        else:

            raise Exception(f"not a valid mode: {self.mode}")


def get_train_dataset(data_dir):
    """Returns data set and data loader for training."""
    HEIGHT = 1920
    WIDTH = 1080

    transform = transforms.Compose(
        [
            transforms.Resize((HEIGHT, WIDTH)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.RandomPerspective(distortion_scale=0.3),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.RandomErasing(scale=(0.02, 0.12)),
        ]
    )

    dataset = CustomDataset(data_dir=data_dir, mode="train", transform=transform)

    return dataset


def get_train_dataloader(dataset, batch_size, num_workers):

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )

    """ @todo random sampler 끼워넣기  """

    return loader


def get_test_dataset(data_dir):
    """Returns data set and data loader for evaluation."""
    HEIGHT = 1920
    WIDTH = 1080

    transform = transforms.Compose(
        [
            transforms.Resize((HEIGHT, WIDTH)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    dataset = CustomDataset(data_dir=data_dir, mode="test", transform=transform)

    return dataset


def get_test_dataloader(dataset, batch_size, num_workers):

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return loader
