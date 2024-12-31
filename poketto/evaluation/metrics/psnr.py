import torch
import torch.distributed as dist
from .base_metric import Metric

class PeakSignalNoiseRatio(Metric):
    def __init__(self):
        self.results = {'pred': [], 'gt': []}
        self.metric_names = ['PSNR']
        self.valid = True
    
    def reset(self):
        for l in self.results.values():
            l.clear()
        self.valid = True

    def fetch(self, result):
        try:
            pred, gt = result['pred'].detach(), result['img']
            if 'norm' in result:
                mean = torch.tensor(result['norm']['mean'], device=pred.device).view(-1, 1, 1)
                std = torch.tensor(result['norm']['std'], device=pred.device).view(-1, 1, 1)
                pred = pred * std + mean
                gt = gt * std + mean
                if not 'minmax' in result:
                    pred = pred / 255.
                    gt = gt / 255.
                pred = pred.clip(0, 1)
                gt = gt.clip(0, 1)
            self.results['pred'].append(pred)
            self.results['gt'].append(gt)
        except:
            for l in self.results.values():
                l.clear()
            self.valid = False

    def compute_metrics(self):
        pred = torch.cat(self.results['pred'])
        gt = torch.cat(self.results['gt'])
        vals = self.calculate(pred, gt)
        metrics = {}
        for i, val in enumerate(vals):
            metrics[self.metric_names[i]] = val.item()
        for k in self.results:
            self.results[k].clear()
        return metrics
    
    @staticmethod
    def calculate(pred: torch.Tensor, gt: torch.Tensor):
        assert pred.size() == gt.size()
        pred = pred.flatten(1)
        gt = gt.flatten(1)
        mse = (gt - pred).square().mean(1)
        eps = 1e-8
        psnr = (-10 * torch.log10(mse + eps)).sum()
        batch_size = torch.tensor(pred.size(0), device=pred.device)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(psnr)
            dist.all_reduce(batch_size)
        psnr = psnr / batch_size
        return [psnr]

    @staticmethod
    def get_best(psnr):
        return max(psnr)