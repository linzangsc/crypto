'''
the entrance of model training
'''

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arguments import trainer_args
from network import crypto_network
from dataset import crypto_dataset

class crypto_trainer:
    def __init__(self, args) -> None:
        self._cfg_file = args.cfg_file
        
        self._param_setup()
        self._model_setup()
        self._data_loader_setup()
        self._loss_function_setup()

    def _param_setup(self):
        with open(self._cfg_file, encoding="utf-8") as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(self.config)

    def _model_setup(self):
        self.model = crypto_network(self.config['feature_dim'], self.config['output_dim'],
                                    self.config['hidden_dim'], self.config['history_window'])
    
    def _data_loader_setup(self):
        self.dataset = crypto_dataset(self.config['root_dir'],
                                      self.config['history_window'], self.config['predict_window'])
        self.data_loader = DataLoader(self.dataset, batch_size=self.config['batch_size'], shuffle=False)

    def _loss_function_setup(self):
        self._classification_loss = nn.BCELoss()
    
    def train(self):
        for epoch in range(self.config['epoch']):
            print(f"epoch: {epoch}")
            for iter, (input_data, label) in enumerate(self.data_loader):
                print(f"iteration: {iter}")
                rise_prediction, fall_prediction = self.model(input_data)
                rise_loss = self._classification_loss(rise_prediction, label[:, 0])
                fall_loss = self._classification_loss(fall_prediction, label[:, 1])

if __name__ == "__main__":
    args = trainer_args().parse()
    trainer = crypto_trainer(args)
    trainer.train()


