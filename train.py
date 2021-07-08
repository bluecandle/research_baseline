import argparse, os, sys
import torch
import numpy as np
import random

from src.trainer import Trainer
from src.utils import load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_fn", type=str, default="train_config")
    parser.add_argument("--data_dir", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_config_fn = args.train_config_fn
    data_dir = args.data_dir
    if not data_dir:
        print(f"data_dir not given: {data_dir}")
        sys.exit()
    elif not os.path.exists(data_dir):
        print(f"data_dir does not exist: {data_dir}")
        sys.exit()

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.join(PROJECT_DIR, "src")
    CONFIG_DIR = os.path.join(SRC_DIR, "config")

    TRAIN_CONFIG_PATH = os.path.join(CONFIG_DIR, "train", f"{train_config_fn}.yml")

    print("TRAIN_CONFIG_PATH", TRAIN_CONFIG_PATH)
    train_config = load_yaml(TRAIN_CONFIG_PATH)

    # SEED
    RANDOM_SEED = train_config["SEED"]["random_seed"]

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        PROJECT_DIR=PROJECT_DIR,
        data_dir=data_dir,
        device=device,
        mode="train",
        config=train_config,
    )
    trainer.train()
