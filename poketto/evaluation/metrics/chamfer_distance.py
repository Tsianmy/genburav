import torch
import torch.distributed as dist
from chamferdist import ChamferDistance as Calculator
from .base_metric import Metric

class ChamferDistance(Metric):
    def __init__(self):
        self.results = {'pred': [], 'gt': []}
        self.metric_names = ['Chamfer_distance']
        self.calculator = Calculator()
        self.valid = True
    
    def reset(self):
        for l in self.results.values():
            l.clear()
        self.valid = True

    def fetch(self, result):
        try:
            self.results['pred'].append(result['pred'].detach())
            self.results['gt'].append(result['gt'])
        except:
            for l in self.results.values():
                l.clear()
            self.valid = False
        
    def compute_metrics(self):
        pred = torch.cat(self.results['pred'])
        target = torch.cat(self.results['gt'])
        vals = self.calculate(pred, target)
        metrics = {}
        for i, val in enumerate(vals):
            metrics[self.metric_names[i]] = val.item()
        for k in self.results:
            self.results[k].clear()
        return metrics
    
    def calculate(self, pred, target):
        assert pred.size() == target.size()
        assert pred.ndim == 2
        batch_size = pred.size(0)
        sum_dis = self.calculator(pred[None, ...], target[None, ...], point_reduction='sum')
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(sum_dis)
            dist.all_reduce(batch_size)
        mean_dis = sum_dis / batch_size * 100
        return [mean_dis]

    @staticmethod
    def get_best(distances):
        return min(distances)