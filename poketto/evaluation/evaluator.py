import torch
from typing import List
from poketto.evaluation import metrics

def metric_wrapper(metric, interval):
    metric.valid = True
    metric.active = True
    metric.interval = interval
    ready = lambda self: self.valid and self.active
    metric.ready = ready.__get__(metric)
    return metric

class Evaluator:
    def __init__(
        self,
        metrics: List[metrics.Metric],
        intervals=1,
        primary_metric=None,
        eval_mode_only=True
    ):
        metrics = metrics if metrics is not None else []
        intervals = [intervals] * len(metrics) if isinstance(intervals, int) else intervals
        assert len(intervals) == len(metrics)
        self.metrics = [
            metric_wrapper(metric, intv) for metric, intv in zip(metrics, intervals)
        ]
        self.best_eval_metrics = {}
        self.primary_metric_key = primary_metric
        self.metric_names = sum([m.metric_names for m in metrics], [])
        self.eval_mode_only = eval_mode_only
        self.training = False
    
    def set_step(self, training, step, num_steps):
        self.training = training
        for metric in self.metrics:
            metric.active = (step == num_steps - 1) or (step % metric.interval == 0)
    
    def reset(self):
        for metric in self.metrics:
            metric.reset()
            metric.valid = True
    
    def ready(self):
        return any(metric.ready() for metric in self.metrics)
    
    @torch.no_grad()
    def update(self, data):
        if self.training and self.eval_mode_only:
            return
        for metric in self.metrics:
            if metric.ready():
                valid = metric.update(data)
                metric.valid = valid

    @torch.no_grad()
    def evaluate(self):
        eval_results = {}
        if self.training and self.eval_mode_only:
            return eval_results
        for metric in self.metrics:
            if not metric.ready():
                continue
            results = metric.get_results()
            eval_results.update(results)

            if not self.training:
                for k, v in results.items():
                    if not k in self.best_eval_metrics:
                        self.best_eval_metrics[k] = v
                    else:
                        self.best_eval_metrics[k] = metric.get_best(
                            [v, self.best_eval_metrics[k]])
                    if self.primary_metric_key is None:
                        self.primary_metric_key = k
        self.reset()

        return eval_results