#! /usr/bin/env python
# -*- coding: utf-8 -*-


import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange


def split_dataset(datadir, ratio=0.1, dataset_type='rfid'):
    """random shuffle train/test set
    """
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, 'spectrum')
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
        random.shuffle(index)

    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')


class Spectrum_dataset(Dataset):
    """Spectrum dataset class.

    Dataset layout (6D CKM):
      - tx_pos.csv  : N_tx rows  (one position per unique TX)
      - rx_pos.csv  : N_rx rows  (one position per unique RX)
      - spectrum/   : N_tx * N_rx PNG files, named XXXXXX.png (1-indexed)
                      Files 1..N_rx belong to TX 0, next N_rx files to TX 1, etc.

    Index mapping (0-based sample_idx from 6-digit filename):
      tx_idx = sample_idx // N_rx
      rx_idx = sample_idx %  N_rx
    """

    def __init__(self, datadir, indexdir, n_rx=50) -> None:
        super().__init__()
        self.datadir = datadir
        self.spectrum_dir = os.path.join(datadir, 'spectrum')
        self.dataset_index = np.loadtxt(indexdir, dtype=str)
        self.n_rx = n_rx  # number of RX positions per TX

        self.tx_pos = pd.read_csv(os.path.join(datadir, 'tx_pos.csv')).values

        rx_pos_path = os.path.join(datadir, 'rx_pos.csv')
        if os.path.exists(rx_pos_path) and os.path.getsize(rx_pos_path) > 0:
            self.rx_pos = pd.read_csv(rx_pos_path).values
            self.has_rx_pos = True
        else:
            self.rx_pos = None
            self.has_rx_pos = False

        self.n_samples = len(self.dataset_index)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        img_name = os.path.join(self.spectrum_dir, self.dataset_index[index] + '.png')
        spectrum = imageio.imread(img_name) / 255.0
        spectrum = torch.tensor(spectrum, dtype=torch.float32)

        sample_idx = int(self.dataset_index[index]) - 1   # 0-based
        tx_idx = sample_idx // self.n_rx
        rx_idx = sample_idx %  self.n_rx

        tx_pos_i = torch.tensor(self.tx_pos[tx_idx], dtype=torch.float32)

        if self.has_rx_pos:
            rx_pos_i = torch.tensor(self.rx_pos[rx_idx], dtype=torch.float32)
        else:
            rx_pos_i = torch.zeros(3, dtype=torch.float32)

        return spectrum, tx_pos_i, rx_pos_i


dataset_dict = {"rfid": Spectrum_dataset}