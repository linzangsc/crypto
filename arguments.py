'''
define arguments here
'''

import argparse

class base_args:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    
    def parse(self):
        self.args = self.parser.parse_args()
        return self.args

class preprocess_args(base_args):
    def __init__(self) -> None:
        super().__init__()
        self.parser.add_argument('--raw_data_path', type=str, help='data root of csv file')
        self.parser.add_argument('--output_path', type=str, help='data root of csv file')
        self.initialized = True

class dataset_args(base_args):
    def __init__(self) -> None:
        super().__init__()
        self.parser.add_argument('--root_dir', type=str, help='data root of csv file')
        self.initialized = True

class network_args(base_args):
    def __init__(self) -> None:
        super().__init__()
        self.initialized = True

class trainer_args(base_args):
    def __init__(self) -> None:
        super().__init__()
        self.parser.add_argument('--cfg_file', default='config.yaml', type=str, help='root of config file')
        self.initialized = True