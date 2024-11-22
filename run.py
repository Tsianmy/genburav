import os
import argparse
import itertools
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['train', 'test'])
    parser.add_argument('--devices', '--devs', type=str, required=True)
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--search_opts', nargs='+')

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
    def split_top_list(values: str):
        if values[0] == '[' and values[-1] == ']':
            valid = True
            for ch in values[1:-1]:
                if ch == '[':
                    break
                elif ch == ']':
                    valid = False
                    break
            if valid:
                values = values[1:-1]
        top_list = []
        start = 0
        for i in range(len(values) + 1):
            if i == len(values) or values[i] == ',':
                if values.count('[', start, i) == values.count(']', start, i):
                    if start < i:
                        top_list.append(values[start:i])
                    start = i + 1
        return top_list

    search, no_search = {}, {}
    for opt in options:
        opt = opt.strip().replace(' ', '')
        key, val = opt.split('=', maxsplit=1)
        val = split_top_list(val)
        if len(val) > 1:
            search[key] = val
        else:
            no_search[key] = val[0]
    no_search = [f'{k}={v}' for k, v in no_search.items()]
    if len(search) > 0:
        for option_comb in itertools.product(*search.values()):
            yield no_search + [f'{k}={v}' for k, v in zip(search.keys(), option_comb)]
    else:
        yield no_search

COM_FIRST_LEN = 8
COM_LEN=3

def trunc(chars, L):
    if len(chars) > L:
        chars = chars[:L] + '_'
    return chars

def option2comment(option):
    key, val = option.split('=', maxsplit=1)
    key_comps = key.split('.')
    for i, comp in enumerate(key_comps):
        key_comps[i] = trunc(comp, COM_FIRST_LEN if i == 0 else COM_LEN)
    key = '.'.join(key_comps)
    return f'{key}={val}'

if __name__ == '__main__':
    args, others = parse_args()
    devices = ','.join([str(int(idx)) for idx in args.devices.split(',')])
    proc_num = (len(devices) + 1) // 2
    cmd = (f"CUDA_VISIBLE_DEVICES={devices} torchrun --nproc_per_node={proc_num} "
           f"--rdzv_backend=static {args.task}.py --cfg='{args.cfg}' ")
    cmd += ' '.join(f"'{a}'" for a in others)
    if args.search_opts:
        for override_options in parse_options(args.search_opts):
            output_dir = os.path.join(
                args.output_dir,
                os.path.splitext(os.path.basename(args.cfg))[0],
                time.strftime("%y%m%d_%H%M%S", time.localtime())
            )
            comments = [option2comment(o) for o in override_options]
            output_dir += '-' + ','.join(comments)
            override_cfg = ' '.join([f"'{o}'" for o in override_options])

            cmd_ = cmd + f" --output_dir '{output_dir}' {override_cfg}"
            print('+', cmd_ + '\n')
            ret = os.system(cmd_)
            if ret != 0:
                break
    else:
        output_dir = os.path.join(
            args.output_dir,
            os.path.splitext(os.path.basename(args.cfg))[0],
            time.strftime("%y%m%d_%H%M%S", time.localtime())
        )
        cmd += f" --output_dir '{output_dir}'"
        print('+', cmd + '\n')
        os.system(cmd)
