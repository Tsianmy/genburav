import torch

class BaseDataPreprocessor:
    def __init__(self, non_blocking=False, **kwargs):
        self._non_blocking = non_blocking
        self._device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    
    def __call__(self, data: dict):
        return NotImplementedError
    
    def to_cuda(self, data: torch.Tensor):
        return data.to(self._device, non_blocking=self._non_blocking)