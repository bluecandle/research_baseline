import os
import time
from datetime import datetime, timezone, timedelta
from .utils import make_directory, get_logger

import torch
import torch.nn as nn

from .data import get_dataloader,

class Trainer(object):
    def __init__(self, PROJECT_DIR, device, mode, config):
        self.PROJECT_DIR = PROJECT_DIR
        self.device = device
        self.mode = mode
        self.config = config

        self.CHECK_POINT = config[self.mode]['check_point']
        self.BATCH_SIZE = config[self.mode]['batch_size']
        self.MODEL = config[self.mode]['model']

        self.KST = timezone(timedelta(hours=9))
        self.TIMESTAMP = datetime.now(tz=self.KST).strftime("%Y%m%d%H%M%S")
        self.TIME_SERIAL = f'{self.MODEL}_{self.TIMESTAMP}'

        # PERFORMANCE RECORD
        self.RESULTS_DIR = os.path.join(self.PROJECT_DIR,"results", self,mode)
        make_directory(self.RESULTS_DIR)
        self.PERFORMANCE_RECORD_DIR = os.path.join(self.RESULTS_DIR, self.TIME_SERIAL)
        make_directory(self.PERFORMANCE_RECORD_DIR)        
        self.PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']
        self.logger = get_logger(name=self.mode, file_path= os.path.join(self.PERFORMANCE_RECORD_DIR, "test_log.log"))


    def train(self):
        """Main training loop."""
        self._logging = True
        self._write_log(f"Training log for {self._timestamp}.\n")

        for key, val in self._args.__dict__.items():
            self._write_log(f"{key}: {val}")

        self._init_train_loader(self._args.csv_path, self._args.batch_size, self._args.num_workers)
        self._init_test_loader(self._args.val_csv_path, self._args.batch_size*2, self._args.num_workers)
        self._init_optimizer()

        for epoch in range(self._args.num_epochs):
            self._write_log(f"\nEpoch {epoch + 1}")
            self._train_epoch()
            self._scheduler.step()
            self.evaluate()
            self._save_model()

    def evaluate(self):
        """Evaluate model performance."""
        self._model.eval()
        if self._test_loader is None:
            self._init_test_loader(self._args.val_csv_path, self._args.batch_size, self._args.num_workers)
        raise NotImplementedError
