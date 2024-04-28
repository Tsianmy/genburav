import os
import time
import datetime
import argparse
import torch
from torch import distributed as dist
from omegaconf import OmegaConf
from poketto.build import (build_model, build_dataloader, build_data_preprocessor,
                         build_evaluator)
from poketto.utils import create_logger, RNGManager

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--checkpoint', type=str)

    args, _ = parser.parse_known_args()

    args.rank = int(os.environ['RANK'])
    args.device_id = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])

    return args

def setup_environment(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.device_id)
    print(f"RANK and WORLD_SIZE: {args.rank}/{args.world_size}")

    logger = create_logger(args.rank, name=__name__)

    ### load config
    config = None
    if args.rank == 0:
        config = OmegaConf.load(args.cfg)
    sync_objs = [config]
    dist.broadcast_object_list(sync_objs, src=0)
    config = sync_objs[0]

    config.merge_with(args.__dict__)
    logger.info(f'Config\n{OmegaConf.to_yaml(config, resolve=True)}')

    ### set seed
    rng_manager = RNGManager()
    rng_manager.set_seed(config.seed)

    return config, logger

def clear_environment():
    dist.destroy_process_group()

def load_model_checkpoint(path_ckpt, config, model):
    with open(path_ckpt, 'rb') as f:
        checkpoint = torch.load(f)
    assert isinstance(checkpoint, dict), f'dict expected, but get {type(checkpoint)}'

    model.load_state_dict(checkpoint['model'])

def test(config, model, val_dataloader, data_preprocessor, evaluator):
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.device_id], broadcast_buffers=False)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {num_params / 1e6}M")

    logger.info("Start evaluating")
    start_time = time.time()

    evaluate(config, model, val_dataloader, data_preprocessor, evaluator)

    total_time = time.time() - start_time
    logger.info(f'Evaluating time {datetime.timedelta(seconds=int(total_time))}')

@torch.no_grad()
def evaluate(config, model, dataloader, data_preprocessor, evaluator):
    model.eval()
    batch_time = 0
    L = len(dataloader)

    end = time.time()
    for it, samples in enumerate(dataloader):
        samples = data_preprocessor(samples, training=False)
        with torch.cuda.amp.autocast(enabled=config.use_amp):
            results = model(samples, mode='predict')

        evaluator.fetch(results)
        
        batch_time += time.time() - end
        end = time.time()

        if it % config.log_interval == 0:
            memory_used = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
            logger.info(
                f'Val [{it}/{L - 1}]  time: {batch_time:.2f}  mem: {memory_used:.0f}MB')
    
    metrics = evaluator.evaluate()
    
    msg = ' *'
    for k, v in metrics.items():
        msg += f' {k}: {v:.3f}'
    logger.info(msg)

    return metrics

if __name__ == '__main__':
    args = parse_args()

    config, logger = setup_environment(args)

    val_dataloader = build_dataloader(config.val_dataloader)

    data_preprocessor = build_data_preprocessor(config.data_preprocessor)

    model = build_model(config.model)
    logger.info(str(model))
    model.cuda()

    if config.checkpoint:
        load_model_checkpoint(config.checkpoint, config, model)

    evaluator = build_evaluator(config.evaluator)

    test(config,
         model,
         val_dataloader,
         data_preprocessor,
         evaluator)
    
    clear_environment()