import torch
from functools import partial

from . import models
from . import optimizers
from . import schedulers
from . import evaluation
from . import visualization
from . import datamodule
from .utils import glogger, _new_instance

def new_model(cfg_model, **kwargs):
    pretrain = cfg_model.pop('pretrain', None)
    model = _new_instance(models, cfg_model, **kwargs)
    if pretrain is not None:
        with open(pretrain, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        glogger.info(f'load pretrained model {pretrain}')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        glogger.info(f'missing keys: {missing}')
        glogger.info(f'unexpected keys: {unexpected}')
        cfg_model['pretrain'] = pretrain

    return model

def new_dataloader(cfg_dataloader, sampler_seed=0):
    dataloader = datamodule.BaseDataloader(**cfg_dataloader, sampler_seed=sampler_seed)

    return dataloader

new_optimizer = partial(_new_instance, optimizers)
new_scheduler = partial(_new_instance, schedulers)
new_evaluator = partial(_new_instance, evaluation)
new_data_preprocessor = partial(_new_instance, datamodule.data_preprocessors)
new_visualizer = partial(_new_instance, visualization)