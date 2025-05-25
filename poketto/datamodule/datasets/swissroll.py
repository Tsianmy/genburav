import torch
from sklearn.datasets import make_swiss_roll
from .base import BaseDataset

class SwissRoll(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def load_data_list(self):
        s_curve, _ = make_swiss_roll(10 ** 4, noise=0.1)
        s_curve = s_curve[:, [0, 2]] / 10.0
        data_list = torch.tensor(s_curve)
        return data_list
    
    def raw_data(self, idx):
        x = self.data_list[idx]
        data = {
            'gt': x,
            'sample_idx': idx
        }
        return data