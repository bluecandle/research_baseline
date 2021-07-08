import os
import time
from datetime import datetime, timezone, timedelta
from .utils import make_directory, get_logger

import torch
from torch import optim
from torch.utils import data as torch_utils_data
from torch.utils.data import DataLoader
import torch.nn as nn

from .data import (
    get_train_dataset,
    get_train_dataloader,
    get_test_dataset,
    get_test_dataloader,
)


class Trainer(object):
    def __init__(self, PROJECT_DIR, data_dir, device, mode, config):
        self.PROJECT_DIR = PROJECT_DIR
        self.data_dir = data_dir
        self.device = device
        self.mode = mode
        self.config = config

        self.KST = timezone(timedelta(hours=9))
        self.TIMESTAMP = datetime.now(tz=self.KST).strftime("%Y%m%d%H%M%S")
        self.TIME_SERIAL = f"{self.MODEL}_{self.TIMESTAMP}"

        # PERFORMANCE RECORD
        self.RESULTS_DIR = os.path.join(self.PROJECT_DIR, "results", self, mode)
        make_directory(self.RESULTS_DIR)
        self.PERFORMANCE_RECORD_DIR = os.path.join(self.RESULTS_DIR, self.TIME_SERIAL)
        make_directory(self.PERFORMANCE_RECORD_DIR)
        self.PERFORMANCE_RECORD_COLUMN_NAME_LIST = config["PERFORMANCE_RECORD"][
            "column_list"
        ]
        self.logger = get_logger(
            name=self.mode,
            file_path=os.path.join(self.PERFORMANCE_RECORD_DIR, "test_log.log"),
        )

        self.VALIDATION_RATIO = None
        if self.mode == "train":
            self.VALIDATION_RATIO = config[self.mode]["validation_ratio"]

        self.BATCH_SIZE = config[self.mode]["batch_size"]
        self.NUM_WORKERS = config[self.mode]["num_workers"]
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None

        self.CHECK_POINT = config[self.mode]["check_point"]
        self.MODEL = config[self.mode]["model"]
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_mean_loss = None
        self.val_mean_loss = None

    def _init_train_validation_data_loader(self):
        original_train_dataset = get_train_dataset(data_dir=self.data_dir)

        original_train_data_len = len(original_train_dataset)
        validation_data_len = int(original_train_data_len * self.VALIDATION_RATIO)
        train_data_len = original_train_data_len - validation_data_len

        train_dataset, validation_dataset = torch_utils_data(
            original_train_dataset, [train_data_len, validation_data_len]
        )
        train_dataloader = get_train_dataloader(
            train_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS
        )
        val_dataloader = get_test_dataloader(
            validation_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS
        )

        return train_dataloader, val_dataloader

    def _init_optimizer(self):

        config_optimizer = self.config[self.mode]["optimizer"]

        if config_optimizer == "Adam":
            self.optimizer = optim.Adam()
        else:
            self.optimizer = optim.SGD()

    def _train_epoch(self):

        pass

    def train(self):
        """Main training loop."""

        if self.logger:
            for key, val in self.config.__dict__.items():
                self.logger.info(f"{key}: {val}")

        self._init_train_validation_data_loader()
        self._init_optimizer()

        for epoch in range(self._args.num_epochs):
            if self.logger:
                self.logger.info(f"\nEpoch {epoch + 1}")

            self._train_epoch()
            self.scheduler.step()
            self.validate()

            self._save_model()

    def validate(self):
        """Evaluate model performance."""
        self.model.eval()
        raise NotImplementedError

    def evaluate(self):
        """Evaluate model performance."""
        self.model.eval()
        raise NotImplementedError
