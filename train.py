'''
the entrance of model training
'''

import time
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
        self._optimizer_setup()
        self._lr_scheduler_setup()

    def _param_setup(self):
        with open(self._cfg_file, encoding="utf-8") as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)
        print(self.config)

    def _model_setup(self):
        self.model = crypto_network(self.config['feature_dim'], self.config['output_dim'],
                                    self.config['hidden_dim'], self.config['history_window'])

    def _data_loader_setup(self):
        print(f"loading data...")
        start_time = time.time()
        self.dataset = crypto_dataset(self.config)
        self.data_loader = DataLoader(self.dataset, batch_size=self.config['batch_size'],
                                      shuffle=False, drop_last=True)
        self._iter_per_epoch = self.dataset.num_samples // self.config['batch_size']
        self._total_step = self._iter_per_epoch * self.config['epoch']
        print(f"data loaded, cost time {time.time() - start_time}")

    def _loss_function_setup(self):
        self._classification_loss = nn.BCELoss()

    def _optimizer_setup(self):
        self._optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['lr'],
                                           weight_decay=self.config['weight_decay'])

    def _lr_scheduler_setup(self):
        self._lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self._optimizer, max_lr=0.01,
                                                                 total_steps=self._total_step)
    
    def _before_train(self):
        print(f"start training")
        training_start_time = time.time()
        return training_start_time

    def _after_train(self, training_start_time):
        print(f"training finish, cost time {time.time() - training_start_time}")

    def train(self):
        training_start_time = self._before_train()
        for epoch in range(self.config['epoch']):
            self._train_before_epoch()
            self._train_in_epoch(epoch)
            self._train_after_epoch()
        self._after_train(training_start_time)

    def _train_before_epoch(self):
        pass

    def _train_in_epoch(self, epoch):
        for iter, (input_data, label) in enumerate(self.data_loader):
            self._train_before_iter()
            losses = self._train_in_iter(input_data, label)
            self._train_after_iter(epoch, iter, losses)

    def _train_after_epoch(self):
        pass

    def _train_before_iter(self):
        pass

    def _train_in_iter(self, input_data, label):
        rise_prediction, fall_prediction = self.model(input_data)
        rise_loss = self._classification_loss(rise_prediction, label[:, 0])
        fall_loss = self._classification_loss(fall_prediction, label[:, 1])
        total_loss = rise_loss + fall_loss

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        return {
            'total_loss': total_loss,
            'rise_loss': rise_loss,
            'fall_loss': fall_loss
        }

    def _train_after_iter(self, epoch, iter, losses):
        if iter % self.config['print_freq'] == 0:
            print(f"epoch: {epoch}, iteration: {iter}/{self._iter_per_epoch}, {losses}")

if __name__ == "__main__":
    args = trainer_args().parse()
    trainer = crypto_trainer(args)
    trainer.train()


