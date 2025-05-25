from torch import optim
SGD = type('SGD', (optim.SGD,), {})
Adam = type('Adam', (optim.Adam,), {})
AdamW = type('AdamW', (optim.AdamW,), {})

_classes = [SGD, Adam, AdamW]

def step_pre_hook(optimizer, args, kwargs):
    kwargs.clear()

def init_wrapper(cls):
    original_init = cls.__init__
    def __init__(self, model, *args, **kwargs):
        params = model.parameters()
        original_init(self, params, *args, **kwargs)
        self.register_step_pre_hook(step_pre_hook)
    cls.__init__ = __init__

for _cls in _classes:
    init_wrapper(_cls)