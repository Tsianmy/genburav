import os
import sys
import random
import numpy as np
import torch
import logging
import copy
from io import TextIOWrapper
from typing import Union, Sequence, Any
from argparse import ArgumentParser, Namespace, Action

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
            logger, fmt='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # init global logger
        if not glogger.hasHandlers():
            configure_logger(glogger,
                             fmt='[%(asctime)s @%(module)s:%(lineno)d] %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')

    return logger

class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val: str) -> Union[int, float, bool, Any]:
        """parse int/float/bool value in the string."""
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val == 'None':
            return None
        return val

    @staticmethod
    def _parse_iterable(val: str) -> Union[list, tuple, Any]:
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple | Any: The expanded list or tuple from the string,
            or single value if no iterable values are found.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]

        if is_tuple:
            return tuple(values)

        return values

    def __call__(self,
                 parser: ArgumentParser,
                 namespace: Namespace,
                 values: Union[str, Sequence[Any], None],
                 option_string: str = None):
        """Parse Variables in string and add them into argparser.

        Args:
            parser (ArgumentParser): Argument parser.
            namespace (Namespace): Argument namespace.
            values (Union[str, Sequence[Any], None]): Argument string.
            option_string (list[str], optional): Option string.
                Defaults to None.
        """
        # Copied behavior from `argparse._ExtendAction`.
        options = copy.copy(getattr(namespace, self.dest, None) or {})
        if values is not None:
            for kv in values:
                key, val = kv.split('=', maxsplit=1)
                options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Tee(TextIOWrapper):
    def __init__(self, textio, filename, mode='w'):
        super().__init__(textio.buffer, textio.encoding, textio.errors,
                         textio.newlines, textio.line_buffering, textio.write_through)
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
        self.avg_loss = AverageMeter()
    
    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        loss = self['loss'].item()
        self.avg_loss.update(loss)
    
    def __str__(self):
        msg = ''
        for k, v in self.items():
            msg += f'  {k}: {v.item():.5f}'
            if k == 'loss':
                msg += f' ({self.avg_loss.avg:.5f})'
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
        return dict(seed=self.seed,
                    random=random.getstate(),
                    numpy=np.random.get_state(),
                    torch=torch.get_rng_state())
    
    def load_state_dict(self, state):
        self.set_seed(state['seed'])
        random.setstate(state['random'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
