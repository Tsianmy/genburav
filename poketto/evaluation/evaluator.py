import torch
from poketto.evaluation import metrics
from typing import List

class Evaluator:
    def __init__(self, metrics: List[metrics.Metric], primary_metric=None):
        self.metrics = metrics or []

        self.best_eval_metrics = {}
        self.primary_metric_key = primary_metric
        self.metric_names = []
        for metric in self.metrics:
            self.metric_names.extend(metric.metric_names)
    
    def reset(self):
        for metric in self.metrics:
            metric.reset()
    
    @torch.no_grad()
    def fetch(self, data):
        for metric in self.metrics:
            if metric.valid:
                metric.fetch(data)

    @torch.no_grad()
    def evaluate(self, training=False):
        eval_results = {}
        for metric in self.metrics:
            if not metric.valid:
                continue
            _result = metric.compute_metrics()
            eval_results.update(_result)

            if not training:
                for k, v in _result.items():
                    if not k in self.best_eval_metrics:
                        self.best_eval_metrics[k] = v
                    else:
                        self.best_eval_metrics[k] = metric.get_best(
                            [v, self.best_eval_metrics[k]])
                    if self.primary_metric_key is None:
                        self.primary_metric_key = k
        self.reset()

        return eval_results