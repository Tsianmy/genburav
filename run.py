import os
import argparse
import itertools
import time
from poketto.utils import DictAction

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['train', 'test'])
    parser.add_argument('--devices', '--devs', type=str, required=True)
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--search_opts', nargs='+', action=DictAction)

    args, unknown = parser.parse_known_args()

    def delete_recur(obj, ch_list: str):
        if isinstance(obj, str):
            for ch in ch_list:
                obj = obj.replace(ch, '')
        elif isinstance(obj, (list, tuple)):
            obj = [delete_recur(x, ch_list) for x in obj]
        return obj

    for k, v in args._get_kwargs():
        setattr(args, k, delete_recur(v, '\'"'))
    
    for i, a in enumerate(unknown):
        unknown[i] = delete_recur(a, '\'"')

    return args, unknown

def parse_options(options):
    search, no_search = {}, {}
    for key, val in options.items():
        if isinstance(val, list):
            search[key] = val
        else:
            no_search[key] = val
    no_search = [f'{k}={v}' for k, v in no_search.items()]
    if len(search) > 0:
        for option_comb in itertools.product(*search.values()):
            yield no_search + [f'{k}={v}' for k, v in zip(search.keys(), option_comb)]
    else:
        yield no_search

if __name__ == '__main__':
    args, others = parse_args()
    devices = ','.join([str(int(idx)) for idx in args.devices.split(',')])
    proc_num = (len(devices) + 1) // 2
    cmd = (f"CUDA_VISIBLE_DEVICES={devices} torchrun --nproc_per_node={proc_num} "
           f"--rdzv_backend=static {args.task}.py --cfg='{args.cfg}' ")
    cmd += ' '.join(f"'{a}'" for a in others)
    if args.search_opts:
        for override_options in parse_options(args.search_opts):
            output_dir = os.path.join(args.output_dir,
                                      os.path.splitext(os.path.basename(args.cfg))[0],
                                      time.strftime("%y%m%d_%H%M%S", time.localtime()))
            output_dir += '-' + ','.join(override_options)
            override_cfg = ' '.join([f"'{o}'" for o in override_options])

            cmd_ = cmd + f" --output_dir '{output_dir}' {override_cfg}"
            print('+', cmd_ + '\n')
            ret = os.system(cmd_)
            if ret != 0:
                break
    else:
        output_dir = os.path.join(args.output_dir,
                                  os.path.splitext(os.path.basename(args.cfg))[0],
                                  time.strftime("%y%m%d_%H%M%S", time.localtime()))
        cmd += f" --output_dir '{output_dir}'"
        print('+', cmd + '\n')
        os.system(cmd)
