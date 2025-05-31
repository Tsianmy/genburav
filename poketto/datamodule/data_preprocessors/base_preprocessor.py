import torch
from poketto.datamodule import batch_augs
from poketto.utils import _new_instance

class BaseDataPreprocessor:
    def __init__(self, non_blocking=False, batch_aug=None, **kwargs):
        self._non_blocking = non_blocking
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        if batch_aug is not None:
            batch_aug = _new_instance(batch_augs, batch_aug)
        self.batch_aug = batch_aug
    
    def __call__(self, data: dict):
        return NotImplementedError
    
    def to_cuda(self, data: torch.Tensor):
        return data.to(self._device, non_blocking=self._non_blocking)