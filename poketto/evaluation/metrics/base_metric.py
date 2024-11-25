class Metric:
    def __init__(self):
        self.metric_names = []
        self.valid = True

    def fetch(self, result):
        raise NotImplementedError
    
    def compute_metrics(self):
        raise NotImplementedError