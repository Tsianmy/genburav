from torch import optim

def step_pre_hook(optimizer, args, kwargs):
    kwargs.clear()

def configure_optimizer(cls):
    original_init = cls.__init__
    def __init__(self, model, *args, **kwargs):
        params = model.parameters()
        original_init(self, params, *args, **kwargs)
        self.register_step_pre_hook(step_pre_hook)
    cls.__init__ = __init__
    return cls

@configure_optimizer
class SGD(optim.SGD): pass
@configure_optimizer
class Adam(optim.Adam): pass
@configure_optimizer
class AdamW(optim.AdamW): pass