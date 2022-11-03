'''
define the class of crypto dataset
usage:
python3.8 dataset.py --root_dir /home/lzh/datasets/crypto_datasets/BTC_hourly/Binance_BTCUSDT_1hour_aft_preprocess5.csv
'''

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from arguments import dataset_args

class crypto_dataset(Dataset):
    def __init__(self, config) -> None:
        self.root_dir = config['root_dir']
        self._history_window = config['history_window']
        self._predict_window = config['predict_window']
        self._feature_dim = config['feature_dim'] #[open, high, low, close, volume, valid]
        self._label_dim = config['label_dim'] #[higher than profit, lower than loss]
        self._profit_limit = config['profit_limit']
        self._loss_limit = config['loss_limit']
        self.num_samples = 0
        self._read_csv_data()
        self._build_shape()
        self._build_samples()

    def __getitem__(self, idx):
        assert idx < self.num_samples
        return [self.npy_data[idx], self.npy_label[idx]]

    def __len__(self):
        return self.num_samples

    def group_tensors(self):
        return [self.npy_data, self.npy_label]

    def _read_csv_data(self):
        self._data_frame = pd.read_csv(self.root_dir)
        data_length = self._data_frame.shape[0]
        self.num_samples = data_length - (self._history_window + self._predict_window) + 1

    def _build_samples(self):
        self.npy_data = np.zeros(self._sample_shape, dtype=np.float32)
        self.npy_label = np.zeros(self._label_shape, dtype=np.float32)
        self._build_features()
        self._build_labels()

    def _build_features(self):
        for i in range(self.num_samples):
            for j in range(self._history_window):
                self.npy_data[i, j, 0] = self._data_frame.iloc[i + j]['open']
                self.npy_data[i, j, 1] = self._data_frame.iloc[i + j]['high']
                self.npy_data[i, j, 2] = self._data_frame.iloc[i + j]['low']
                self.npy_data[i, j, 3] = self._data_frame.iloc[i + j]['close']
                self.npy_data[i, j, 4] = self._data_frame.iloc[i + j]['Volume USDT']
                self.npy_data[i, j, 5] = self._data_frame.iloc[i + j]['valid']

    def _build_labels(self):
        for i in range(self.num_samples):
            buy_price = self._data_frame.iloc[i + self._history_window]['close']
            profit_price = buy_price * (1. + self._profit_limit)
            loss_price = buy_price * (1. - self._loss_limit)
            for j in range(self._predict_window):
                if self.npy_label[i, 0] and self.npy_label[i, 1]:
                    break
                if not self._data_frame.iloc[i + self._history_window + j]['valid']:
                    break
                if self._data_frame.iloc[i + self._history_window + j]['high'] > profit_price:
                    self.npy_label[i, 0] = 1
                if self._data_frame.iloc[i + self._history_window + j]['low'] < loss_price:
                    self.npy_label[i, 1] = 1

    def _build_shape(self):
        self._sample_shape = (self.num_samples, self._history_window, self._feature_dim)
        self._label_shape = (self.num_samples, self._label_dim)

if __name__ == "__main__":
    args = dataset_args().parse()
    root_dir = args.root_dir
    assert root_dir, "please input root dir of csv file"
    dataset = crypto_dataset(root_dir, 3, 3)
    data_and_label = dataset.group_tensors()
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    for epoch in range(100):
        for iter, data in enumerate(data_loader):
            print(iter)