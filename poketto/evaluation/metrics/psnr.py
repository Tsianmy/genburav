import torch
import torch.distributed as dist
from .base_metric import Metric

class PeakSignalNoiseRatio(Metric):
    def __init__(self):
        self.metric_names = ['PSNR']
        self.reset()
    
    def reset(self):
        self.psnr_sum = 0
        self.num_samples = 0

    def update(self, data) -> bool:
        try:
            pred, gt = data['pred'].detach(), data['img'].detach()
        except:
            self.reset()
            return False
        if 'norm' in data:
            mean = torch.tensor(data['norm']['mean'], device=pred.device).view(-1, 1, 1)
            std = torch.tensor(data['norm']['std'], device=pred.device).view(-1, 1, 1)
            pred = pred * std + mean
            gt = gt * std + mean
            if not 'minmax' in data:
                pred = pred / 255.
                gt = gt / 255.
            pred = pred.clip(0, 1)
            gt = gt.clip(0, 1)
        psnr_batch_sum = self.calculate(pred, gt)
        self.psnr_sum += psnr_batch_sum
        self.num_samples += pred.shape[0]
        return True

    def get_results(self):
        metrics = {}
        psnr_sum, num_samples = self.psnr_sum.clone(), self.num_samples
        num_samples = torch.tensor(num_samples, device=psnr_sum.device)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(psnr_sum)
            dist.all_reduce(num_samples)
        vals = [psnr_sum / num_samples]
        for i, val in enumerate(vals):
            metrics[self.metric_names[i]] = val.item()
        return metrics
    
    @staticmethod
    def calculate(pred: torch.Tensor, gt: torch.Tensor):
        assert pred.shape == gt.shape
        pred = pred.flatten(1)
        gt = gt.flatten(1)
        mse = (gt - pred).square().mean(1)
        eps = 1e-8
        psnr_sum = (-10 * torch.log10(mse + eps)).sum()
        return psnr_sum

    @staticmethod
    def get_best(psnr):
        return max(psnr)