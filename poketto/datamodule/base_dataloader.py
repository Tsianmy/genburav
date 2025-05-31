from typing import Iterable, List, Callable, Dict
import torch
from torch.utils.data.sampler import Sampler
from poketto import datamodule
from poketto.utils import _new_instance
from .utils import get_collate_fn

Config = Dict

class BaseDataloader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset: Config,
        batch_size: int | None = 1,
        shuffle: bool | None = None,
        sampler: Config = None,
        batch_sampler: Sampler[List] | Iterable[List] | None = None,
        sampler_seed: int = 0,
        num_workers: int = 0,
        collate_keys: List = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Callable[[int], None] | None = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory_device: str = ""
    ):
        cfg_dataset = dataset
        dataset = _new_instance(datamodule.datasets, cfg_dataset)

        cfg_sampler = sampler
        if cfg_sampler is not None:
            cfg_sampler['seed'] = cfg_sampler.get('seed', sampler_seed)
            sampler = _new_instance(datamodule.samplers, cfg_sampler, dataset)
        else:
            sampler = None

        if collate_keys is None:
            collate_keys = []
        collate_fn = get_collate_fn(collate_keys)

        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, pin_memory_device=pin_memory_device)