from typing import Sequence
from poketto import factory

class Evaluator:
    def __init__(self, metrics, primary_metric=None):
        if not isinstance(metrics, Sequence):
            metrics = [metrics]
        self.metrics = []
        for metric in metrics:
            self.metrics.append(factory.new_metric(metric))

        self.best_eval_metrics = {}
        self.primary_metric_key = primary_metric
    
    def fetch(self, data):
        for metric in self.metrics:
            metric.fetch(data)

    def evaluate(self):
        eval_results = {}
        for metric in self.metrics:
            _result = metric.compute_metrics()
            eval_results.update(_result)

            for k, v in _result.items():
                if not k in self.best_eval_metrics:
                    self.best_eval_metrics[k] = v
                else:
                    self.best_eval_metrics[k] = metric.get_best(
                        [v, self.best_eval_metrics[k]])
                if self.primary_metric_key is None:
                    self.primary_metric_key = k
                

        return eval_results