import torch

class Metric:
    def __init__(self):
        self.metric_names = []

    def reset(self):
        pass

    @torch.inference_mode()
    def update(self, data) -> bool:
        raise NotImplementedError
    
    @torch.inference_mode()
    def get_results(self):
        raise NotImplementedError