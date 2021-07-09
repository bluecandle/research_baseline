from logging import critical
import os
import time
from datetime import datetime, timezone, timedelta
from .utils import make_directory, get_logger, save_yaml

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data as torch_utils_data
import torch.nn as nn

from .data import (
    get_train_dataset,
    get_train_dataloader,
    get_test_dataset,
    get_test_dataloader,
)
from .earlystoppers import LossEarlyStopper


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
            self.learning_rate = config[self.mode]['learning_rate']
            self.early_stopping_patience = config[self.mode]['early_stopping_patience']

        self.BATCH_SIZE = config[self.mode]["batch_size"]
        self.NUM_WORKERS = config[self.mode]["num_workers"]
        self.train_dataloader = None 
        self.validation_dataloader = None
        self.test_dataloader = None

        self.CHECK_POINT = config[self.mode]["check_point"]
        self.NUM_EPOCHS = config[self.mode]["num_epochs"]
        self.MODEL = config[self.mode]["model"]
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_mean_loss = None
        self.val_mean_loss = None
        self.criterion = 1E+8

        self._init_train_validation_data_loader()
        self._init_optimizer()
        self._init_scheduler()

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
        validation_dataloader = get_test_dataloader(
            validation_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS
        )

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader 

    def _init_model(self):

        # model = 

        model_init_state_dict = None
        if self.CHECK_POINT:

            if torch.load(self.CHECK_POINT).get("model_state_dict") is not None:
                model_init_state_dict = torch.load(self.CHECK_POINT)["model_state_dict"]
            elif torch.load(self.CHECK_POINT).get("model_dict") is not None:
                model_init_state_dict = torch.load(self.CHECK_POINT)["model_dict"]
            elif torch.load(self.CHECK_POINT).get("state_dict") is not None:
                model_init_state_dict = torch.load(self.CHECK_POINT)["state_dict"]
            else:
                model_init_state_dict = torch.load(self.CHECK_POINT)

            # model_init_state_dict.pop(".weight")

        if model_init_state_dict:
            # model.load_state_dict(model_init_state_dict, strict=False)

            pass

        # self.model =

        raise NotImplementedError 

    def _init_optimizer(self):

        config_optimizer = self.config[self.mode]["optimizer"]

        if config_optimizer == "Adam":
            optimizer = optim.Adam()
        else:
            optimizer = optim.SGD()

        optimizer_init_state_dict = None
        if self.CHECK_POINT:

            if torch.load(self.CHECK_POINT).get("optimizer_state_dict") is not None:
                optimizer_init_state_dict = torch.load(self.CHECK_POINT)["optimizer_state_dict"]

        if optimizer_init_state_dict:
            ret = optimizer.load_state_dict(optimizer_init_state_dict, strict = False)
            msg = f"[optimizer] |{ret}"
            print(msg)
            if self.logger:
                self.logger.info(msg)

        self.optimizer = optimizer

    def _init_scheduler(self):

        config_scheduler = self.config[self.mode]["scheduler"]
        
        # if config_scheduler == "":
        #     self.scheduler = lr_scheduler.
        # else:
        #     self.scheduler = lr_scheduler.
        scheduler = lr_scheduler.OneCycleLR(optimizer=self.optimizer, pct_start=0.1, div_factor=1e5, max_lr=self.learning_rate, epochs=self.NUM_EPOCHS, steps_per_epoch=len(self.train_dataloader))

        scheduler_init_state_dict = None
        if self.CHECK_POINT:

            if torch.load(self.CHECK_POINT).get("scheduler_state_dict") is not None:
                scheduler_init_state_dict = torch.load(self.CHECK_POINT)["scheduler_state_dict"]

        if scheduler_init_state_dict:
            
            ret = scheduler.load_state_dict(scheduler_init_state_dict, strict = False)
            msg = f"[scheduler] |{ret}"
            print(msg)
            if self.logger:
                self.logger.info(msg)

        self.scheduler = scheduler

    def _train_epoch(self, epoch):
        """Train model."""
        self.model.train()

        # self.train_mean_loss = 
        raise NotImplementedError

    def _validate_epoch(self, epoch):
        """Validate model performance."""
        self.model.eval()

        # self.val_mean_loss = 
        raise NotImplementedError

    def _save_model(self,epoch):

        if self.val_mean_loss < self.criterion:
            self.criterion = self.val_mean_loss
            save_path = os.path.join(self.PERFORMANCE_RECORD_DIR, f'best_{epoch+1}.pt')
            check_point = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
            }
            torch.save(check_point, save_path)
            msg = f"Model saved: {save_path}"
            print(msg)
            if self.logger:
                self.logger.info(msg)

    def train(self):
        train_start_time = time.time()
        if self.logger:
            for key, val in self.config.__dict__.items():
                self.logger.info(f"{key}: {val}")

        early_stopper = LossEarlyStopper(patience=self.early_stopping_patience, verbose=True, logger=self.logger)

        save_yaml(os.path.join(self.PERFORMANCE_RECORD_DIR, 'train_config.yaml'), self.config)
        
        """Main training loop."""
        for epoch in range(self.NUM_EPOCHS):
            start_time = time.time()
            start_msg = f"############# Strat Epoch: {epoch+1} #################"

            print(start_msg)
            if self.logger:
                self.logger.info(start_msg)
            
            self._train_epoch(epoch)
            self.scheduler.step()
            self._validate_epoch(epoch)
            self._save_model(epoch)

            time_per_epoch = (time.time() - start_time)/60
        
            end_msg = f"############# Finished Epoch: {epoch+1} | Took {time_per_epoch:.2f} mins #################"
            print(end_msg)
            if self.logger:
                self.logger.info(end_msg)

            try:
            # early_stopping check
                early_stopper.check_early_stopping(loss=self.val_mean_loss)
            except Exception as e:
                msg = f"error with early_stopper\n{e}"
                print(msg)
                if self.logger:
                    self.logger.info(msg)

            if early_stopper.stop:
                early_stop_msg = '############## Early stopped ##############'
                print(early_stop_msg)
                if self.logger:
                    self.logger.info(early_stop_msg)
                    break
        
        total_train_time = time.time()-train_start_time
        msg = f"Total Training Time: {total_train_time}"
        print(msg)
        if self.logger:
            self.logger.info(msg)

    def evaluate(self):
        """Evaluate model performance."""
        self.model.eval()

        raise NotImplementedError
