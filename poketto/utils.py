import os
import sys
import random
import numpy as np
import torch
import logging
from io import TextIOWrapper
from timm.utils import AverageMeter

glogger = logging.getLogger('global')

def configure_logger(logger, fmt, datefmt=None):
    console_handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(fmt, datefmt)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

def create_logger(dist_rank=0, name=None):
    logger = logging.getLogger(name)

    if dist_rank == 0:
        configure_logger(
            logger, fmt='[%(asctime)s %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # init global logger
        if not glogger.hasHandlers():
            configure_logger(
                glogger,
                fmt='[%(asctime)s @%(module)s:%(lineno)d %(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

    return logger

class Tee(TextIOWrapper):
    def __init__(self, textio, filename, mode='w'):
        super().__init__(
            textio.buffer,
            textio.encoding,
            textio.errors,
            textio.newlines,
            textio.line_buffering,
            textio.write_through
        )
        self.__dict__.update(textio.__dict__)
        self.file = open(filename, mode, buffering=1)
    
    def write(self, s):
        super().write(s)
        self.file.write(s)
    
    def writelines(self, _lines):
        super().writelines(_lines)
        self.file.writelines(_lines)
    
    def flush(self):
        super().flush()
        self.file.flush()

    def close(self):
        super().close()
        self.file.close()

class LossDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avgs = {}
    
    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        for k in self:
            if not k in self.avgs:
                self.avgs[k] = AverageMeter()
            self.avgs[k].update(self[k].item())
    
    def __str__(self):
        msg = ''
        for k, v in self.items():
            msg += f'  {k}: {v.item():.5f} ({self.avgs[k].avg:.5f})'
        return msg.lstrip()

class RNGManager:
    seed = 0
    def set_seed(self, seed):
        self.seed = seed
        os.environ['RANDOM_SEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        glogger.info(f'Set Seed {seed}')
    
    def state_dict(self):
        torch_state = dict(cpu=torch.get_rng_state())
        if torch.cuda.is_available():
            torch_state['cuda'] = torch.cuda.get_rng_state()
        return dict(
            seed=self.seed,
            random=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch_state
        )
    
    def load_state_dict(self, state):
        self.set_seed(state['seed'])
        random.setstate(state['random'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch']['cpu'])
        if 'cuda' in state['torch']:
            torch.cuda.set_rng_state_all([state['torch']['cuda']] * torch.cuda.device_count())
