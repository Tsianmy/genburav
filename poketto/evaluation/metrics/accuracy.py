import torch
from torch import distributed as dist
from .base_metric import Metric

class Accuracy(Metric):
    def __init__(self, topk = (1, )):
        self.results = {'pred': [], 'gt_label': []}
        if isinstance(topk, int):
            self.topk = (topk, )
        else:
            self.topk = tuple(topk)
        self.metric_names = [f'top{i}' for i in self.topk]
        
    def fetch(self, result):
        self.results['pred'].append(result['pred'])
        self.results['gt_label'].append(result['gt_label'])
    
    def compute_metrics(self):
        pred = torch.cat(self.results['pred'])
        target = torch.cat(self.results['gt_label'])
        if len(target.shape) > 1:
            target = target.argmax(1)
        acc = self.calculate(pred, target, self.topk)
        metrics = {}
        for i, v in enumerate(acc):
            metrics[self.metric_names[i]] = v.item()
        
        for l in self.results:
            self.results[l].clear()
        return metrics

    @staticmethod
    def calculate(pred, target, topk):
        batch_size = pred.size(0)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            # For pred label, ignore topk and acc
            pred_label = pred.int()
            correct = pred.eq(target).sum(0)
            batch_size = torch.tensor([batch_size], device=correct.device)
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(correct)
                dist.all_reduce(batch_size)
            acc = correct.float() * 100. / batch_size
            return [acc]
        else:
            # For pred score, calculate on all topk and thresholds.
            pred = pred.float()
            maxk = min(max(topk), pred.size()[1])

            pred_score, pred_label = pred.topk(maxk, dim=1)
            pred_label = pred_label.t()
            correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
            correct_k = torch.zeros(len(topk), dtype=torch.int, device=correct.device)
            for i, k in enumerate(topk):
                correct_k[i] = correct[:min(k, maxk)].reshape(-1).sum(0)
            batch_size = torch.tensor([batch_size], device=correct.device)

            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(correct_k)
                dist.all_reduce(batch_size)
            acc = correct_k.float() * 100. / batch_size
            return acc
    
    @staticmethod
    def get_best(accs):
        return max(accs)