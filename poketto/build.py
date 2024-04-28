import torch
import torch.utils.data as torchdata

from . import models
from . import optimizers
from . import schedulers
from . import evaluation
from . import datasets
from .datasets import transforms, samplers, data_preprocessors, batch_augs
from .evaluation import metrics
from .utils import glogger

def _general_build(module, cfg_obj, *args, **kwargs):
    type_obj = cfg_obj.pop('type')
    cls_obj = getattr(module, type_obj)
    obj = cls_obj(*args, **cfg_obj, **kwargs)
    cfg_obj['type'] = type_obj

    return obj

def build_model(cfg_model):
    pretrain = cfg_model.pop('pretrain', None)
    model = _general_build(models, cfg_model)
    if pretrain is not None:
        state_dict = torch.load(pretrain, map_location='cpu')
        glogger.info(f'load pretrained model {pretrain}')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        glogger.info(f'missing keys: {missing}')
        glogger.info(f'unexpected keys: {unexpected}')

    return model

def build_transform(cfg_transform):
    transform = _general_build(transforms, cfg_transform)
    
    return transform

def build_dataset(cfg_dataset):
    dataset = _general_build(datasets, cfg_dataset)

    return dataset

def build_sampler(cfg_sampler, dataset):
    sampler = _general_build(samplers, cfg_sampler, dataset)

    return sampler

def build_dataloader(cfg_dataloader, sampler_seed=None):
    cfg_dataset = cfg_dataloader.pop('dataset')
    dataset = build_dataset(cfg_dataset)

    cfg_sampler = cfg_dataloader.pop('sampler', None)
    cfg_sampler['seed'] = cfg_sampler.get('seed', sampler_seed)
    sampler = build_sampler(cfg_sampler, dataset) if cfg_sampler is not None else None

    collate_keys = cfg_dataloader.pop('collate_keys', [])
    collate_fn = datasets.get_collate_fn(collate_keys)
    dataloader = torchdata.DataLoader(dataset, sampler=sampler, collate_fn=collate_fn,
                                      **cfg_dataloader)
    cfg_dataloader['dataset'] = cfg_dataset

    return dataloader

def build_optimizer(cfg_optim, params):
    optimizer = _general_build(optimizers, cfg_optim, params)

    return optimizer

def build_scheduler(cfg_scheduler, optimizer, epoch_len=None):
    scheduler = _general_build(schedulers, cfg_scheduler, optimizer, epoch_len=epoch_len)

    return scheduler

def build_evaluator(cfg_evaluator):
    evaluator = _general_build(evaluation, cfg_evaluator)
    
    return evaluator

def build_metric(cfg_metric):
    metric = _general_build(metrics, cfg_metric)
    
    return metric

def build_data_preprocessor(cfg_data_preprocessor):
    data_preprocessor = _general_build(data_preprocessors, cfg_data_preprocessor)

    return data_preprocessor

def build_batch_aug(cfg_batch_aug):
    batch_augment = _general_build(batch_augs, cfg_batch_aug)

    return batch_augment