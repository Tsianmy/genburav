import torch
import torch.distributed as dist
from chamferdist import ChamferDistance as Calculator
from .base_metric import Metric

class ChamferDistance(Metric):
    def __init__(self):
        self.data = {'pred': [], 'gt': []}
        self.metric_names = ['Chamfer_distance']
    
    def reset(self):
        for l in self.data.values():
            l.clear()

    def update(self, data) -> bool:
        try:
            self.data['pred'].append(data['pred'].detach())
            self.data['gt'].append(data['gt'].detach())
        except:
            self.reset()
            return False
        return True
        
    @torch.inference_mode()
    def get_results(self):
        pred = torch.cat(self.data['pred'])
        target = torch.cat(self.data['gt'])
        vals = self.calculate(pred, target)
        metrics = {}
        for i, val in enumerate(vals):
            metrics[self.metric_names[i]] = val.item()
        return metrics
    
    @staticmethod
    @torch.inference_mode()
    def calculate(pred, target):
        calculator = Calculator()
        assert pred.size() == target.size()
        assert pred.ndim == 2
        if dist.is_initialized() and dist.get_world_size() > 1:
            world_size = dist.get_world_size()
            pred_list = [torch.zeros_like(pred) for _ in range(world_size)]
            target_list = [torch.zeros_like(target) for _ in range(world_size)]
            dist.all_gather(pred_list, pred)
            dist.all_gather(target_list, target)
            pred = torch.cat(pred_list)
            target = torch.cat(target_list)
        sum_dis = calculator(pred[None, ...], target[None, ...], point_reduction='sum')
        batch_size = pred.size(0)
        mean_dis = sum_dis / batch_size * 100
        return [mean_dis]

    @staticmethod
    def get_best(distances):
        return min(distances)