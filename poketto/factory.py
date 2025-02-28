import torch
import torch.utils.data as torchdata

from . import models
from . import optimizers
from . import schedulers
from . import evaluation
from . import visualization
from . import datasets
from .datasets import transforms, samplers, data_preprocessors, batch_augs
from .evaluation import metrics
from .utils import glogger

def _new_obj(module, cfg_obj, *args, **kwargs):
    type_obj = cfg_obj.pop('type')
    cls_obj = getattr(module, type_obj)
    obj = cls_obj(*args, **cfg_obj, **kwargs)
    cfg_obj['type'] = type_obj

    return obj

def new_model(cfg_model):
    pretrain = cfg_model.pop('pretrain', None)
    model = _new_obj(models, cfg_model)
    if pretrain is not None:
        with open(pretrain, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        glogger.info(f'load pretrained model {pretrain}')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        glogger.info(f'missing keys: {missing}')
        glogger.info(f'unexpected keys: {unexpected}')

    return model

def new_transform(cfg_transform):
    transform = _new_obj(transforms, cfg_transform)
    
    return transform

def new_dataset(cfg_dataset):
    cfg_transforms = cfg_dataset.pop('transforms', [])
    transforms = [new_transform(cfg_tr) for cfg_tr in cfg_transforms]
    dataset = _new_obj(datasets, cfg_dataset, transforms=transforms)

    return dataset

def new_sampler(cfg_sampler, dataset):
    sampler = _new_obj(samplers, cfg_sampler, dataset)

    return sampler

def new_dataloader(cfg_dataloader, sampler_seed=0):
    cfg_dataset = cfg_dataloader.pop('dataset')
    dataset = new_dataset(cfg_dataset)

    cfg_sampler = cfg_dataloader.pop('sampler', None)
    if cfg_sampler is not None:
        cfg_sampler['seed'] = cfg_sampler.get('seed', sampler_seed)
        sampler = new_sampler(cfg_sampler, dataset)
    else:
        sampler = None

    collate_keys = cfg_dataloader.pop('collate_keys', [])
    collate_fn = datasets.get_collate_fn(collate_keys)
    dataloader = torchdata.DataLoader(
        dataset, sampler=sampler, collate_fn=collate_fn, **cfg_dataloader)
    cfg_dataloader['dataset'] = cfg_dataset

    return dataloader

def new_optimizer(cfg_optim, params):
    optimizer = _new_obj(optimizers, cfg_optim, params)

    return optimizer

def new_scheduler(cfg_scheduler, optimizer, epoch_len=None):
    scheduler = _new_obj(schedulers, cfg_scheduler, optimizer, epoch_len=epoch_len)

    return scheduler

def new_metric(cfg_metric):
    metric = _new_obj(metrics, cfg_metric)
    
    return metric

def new_evaluator(cfg_evaluator):
    cfg_metrics = cfg_evaluator.pop('metrics', [])
    metrics = [new_metric(cfg_metric) for cfg_metric in cfg_metrics]
    evaluator = _new_obj(evaluation, cfg_evaluator, metrics=metrics)
    
    return evaluator

def new_batch_aug(cfg_batch_aug):
    batch_augment = _new_obj(batch_augs, cfg_batch_aug)

    return batch_augment

def new_data_preprocessor(cfg_data_preprocessor):
    cfg_batch_aug = cfg_data_preprocessor.pop('batch_aug', None)
    batch_aug = new_batch_aug(cfg_batch_aug) if cfg_batch_aug is not None else None
    data_preprocessor = _new_obj(
        data_preprocessors, cfg_data_preprocessor, batch_aug=batch_aug)

    return data_preprocessor

def new_visualizer(cfg_visualizer, save_dir, **kwargs):
    visualizer = _new_obj(visualization, cfg_visualizer, save_dir=save_dir, **kwargs)

    return visualizer