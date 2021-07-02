import argparse, os
import torch
import numpy as np
import random

from src.trainer import Trainer
from src.utils import load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_config_fn", type=str, default="test_config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_config_fn = args.test_config_fn

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.join(PROJECT_DIR,"src")
    RESULTS_DIR = os.path.join(PROJECT_DIR, "results") 
    CONFIG_DIR = os.path.join(SRC_DIR,"config")

    DIRS = {
        "PROJECT_DIR" : PROJECT_DIR,
        "SRC_DIR" : SRC_DIR,
        "RESULTS_DIR" : RESULTS_DIR,
        "CONFIG_DIR" : CONFIG_DIR,
    }

    TEST_CONFIG_PATH = os.path.join(CONFIG_DIR, "test", f'{test_config_fn}.yml')

    print("TEST_CONFIG_PATH",TEST_CONFIG_PATH)
    test_config = load_yaml(TEST_CONFIG_PATH)

    # SEED
    RANDOM_SEED = test_config['SEED']['random_seed']

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    trainer = Trainer(mode = "train", config = test_config, dirs = DIRS)
    trainer.evaluate()