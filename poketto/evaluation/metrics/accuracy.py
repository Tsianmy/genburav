import torch
from torch import distributed as dist
from .base_metric import Metric

class Accuracy(Metric):
    def __init__(self, topk = (1, )):
        if isinstance(topk, int):
            self.topk = (topk, )
        else:
            self.topk = tuple(topk)
        self.metric_names = [f'top{i}' for i in self.topk]
        self.reset()
    
    def reset(self):
        self.num_correct = 0
        self.num_samples = 0

    @torch.inference_mode()
    def update(self, data) -> bool:
        pred, target = data['pred'].detach(), data['gt_label'].detach()
        if pred.ndim > 2:
            pred = pred.view(-1, pred.shape[-1])
        if target.ndim > 1:
            target = target.view(-1, target.shape[-1]).argmax(dim=-1)
        correct = self.calculate(pred, target, self.topk)
        self.num_correct += correct
        self.num_samples += pred.shape[0]
        return True
    
    @torch.inference_mode()
    def get_results(self):
        metrics = {}
        correct, num_samples = self.num_correct.clone(), self.num_samples
        num_samples = torch.tensor(num_samples, device=correct.device)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(correct)
            dist.all_reduce(num_samples)
        accs = correct * 100. / num_samples
        for i, v in enumerate(accs):
            metrics[self.metric_names[i]] = v.item()
        
        return metrics

    @staticmethod
    @torch.inference_mode()
    def calculate(pred, target, topk):
        assert pred.shape[0] == target.shape[0], \
            f"The size of pred ({pred.shape[0]}) doesn't match "\
            f'the target ({target.shape[0]}).'

        if pred.ndim == 1:
            # For pred label, ignore topk and acc
            pred_label = pred.int()
            correct = pred.eq(target).sum(0).view(-1)
            return correct
        else:
            # For pred score, calculate on all topk and thresholds.
            pred = pred.float()
            maxk = min(max(topk), pred.shape[1])

            pred_score, pred_label = pred.topk(maxk, dim=-1)
            pred_label = pred_label.t()
            correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
            correct_k = torch.zeros(len(topk), dtype=torch.int, device=correct.device)
            for i, k in enumerate(topk):
                correct_k[i] = correct[:min(k, maxk)].reshape(-1).sum(0)
            return correct_k
    
    @staticmethod
    def get_best(accs):
        return max(accs)