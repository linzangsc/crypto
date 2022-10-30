'''
the entrance of model training
'''

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from arguments import trainer_args
from network import crypto_network
from dataset import crypto_dataset

class crypto_trainer:
    def __init__(self, args) -> None:
        self._cfg_file = args.cfg_file
        
        self._param_setup()
        self._model_setup()
        # self._data_loader_setup()
        self._loss_function_setup()

    def _param_setup(self):
        with open(self._cfg_file, encoding="utf-8") as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)
        self._root_dir = self.config['root_dir']
        self._history_window = self.config['history_window']
        self._predict_window = self.config['predict_window']

        self._feature_dim = self.config['feature_dim']
        self._output_dim = self.config['output_dim']
        self._hidden_dim = self.config['hidden_dim']

    def _model_setup(self):
        self.model = crypto_network(self._feature_dim, self._output_dim,
                                    self._hidden_dim, self._history_window)
    
    def _data_loader_setup(self):
        self.dataset = crypto_dataset(self._root_dir,
                                      self._history_window, self._predict_window)

    def _loss_function_setup(self):
        self._classification_loss = nn.BCELoss()
    
    def train(self):
        pass

if __name__ == "__main__":
    args = trainer_args().parse()
    trainer = crypto_trainer(args)


