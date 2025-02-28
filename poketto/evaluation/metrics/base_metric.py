class Metric:
    def __init__(self):
        self.metric_names = []

    def reset(self):
        pass

    def update(self, data) -> bool:
        raise NotImplementedError
    
    def get_results(self):
        raise NotImplementedError