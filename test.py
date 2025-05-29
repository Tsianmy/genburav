import os
import sys
import time
import datetime
import argparse
import torch
from torch import distributed as dist
from omegaconf import OmegaConf
from poketto import factory
from poketto.utils import create_logger, RNGManager, Tee

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--log_freq', type=int, default=2)

    args, _ = parser.parse_known_args()

    args.output_dir = os.path.join(args.output_dir, 'test')
    args.rank = int(os.environ['RANK'])
    args.device_id = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])

    return args

def setup_environment(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.device_id)
    print(f"RANK and WORLD_SIZE: {args.rank}/{args.world_size}")

    if args.logging:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(
            args.output_dir, f'{time.strftime("%y%m%d_%H%M%S", time.localtime())}.txt'
        )
        tee = Tee(sys.stdout, logfile, mode='a')
        sys.stdout, sys.stderr = tee, tee
    logger = create_logger(args.rank, name=__name__)

    ### load config
    config = OmegaConf.load(args.cfg)
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
        checkpoint = torch.load(f, map_location='cpu', weights_only=False)
    assert isinstance(checkpoint, dict), f'dict expected, but get {type(checkpoint)}'

    model.load_state_dict(checkpoint['model'])

def test(config, model, val_dataloader, data_preprocessor, evaluator, visualizer=None):
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.device_id], broadcast_buffers=False)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {num_params / 1e6}M")

    logger.info("Start evaluating")
    start_time = time.time()

    evaluate(config, model, val_dataloader, data_preprocessor, evaluator, visualizer)

    total_time = time.time() - start_time
    logger.info(f'Evaluating time {datetime.timedelta(seconds=int(total_time))}')

@torch.inference_mode()
def evaluate(config, model, dataloader, data_preprocessor, evaluator, visualizer=None):
    model.eval()
    batch_time = 0
    L = len(dataloader)
    log_interval = max(L // config.log_freq, 1)

    end = time.time()
    for it, samples in enumerate(dataloader):
        samples = data_preprocessor(samples, training=False)
        with torch.amp.autocast('cuda', enabled=config.use_amp):
            results = model(samples)

        evaluator.update(results)
        if visualizer is not None:
            visualizer.save(results, it * dataloader.batch_size)
        
        batch_time += time.time() - end
        end = time.time()

        if it % log_interval == 0:
            memory_used = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
            logger.info(
                f'Val [{it}/{L - 1}]  time: {batch_time:.2f}  mem: {memory_used:.0f}MB')
    
    metrics = evaluator.evaluate()
    
    msg = ' *'
    for k, v in metrics.items():
        msg += f' {k}: {v:.3f}'
    logger.info(msg)

if __name__ == '__main__':
    args = parse_args()

    config, logger = setup_environment(args)

    logger.info(f'building dataloader')
    val_dataloader = factory.new_dataloader(config.val_dataloader)

    logger.info(f'building data_preprocessor {config.data_preprocessor["type"]}')
    data_preprocessor = factory.new_data_preprocessor(config.data_preprocessor)

    logger.info(f'building model {config.model["type"]}')
    model = factory.new_model(config.model, dataset=val_dataloader.dataset)

    if config.checkpoint:
        load_model_checkpoint(config.checkpoint, config, model)

    logger.info(str(model))
    model.cuda()

    logger.info(f'building evaluator')
    evaluator = factory.new_evaluator(config.evaluator)

    visualizer = None
    if config.rank == 0 and config.logging and getattr(config, 'visualizer', None) is not None:
        logger.info(f'building visualizer {config.visualizer["type"]}')
        config.visualizer['use_tensorboard'] = False
        visualizer = factory.new_visualizer(config.visualizer, config.output_dir)

    test(
        config,
        model,
        val_dataloader,
        data_preprocessor,
        evaluator,
        visualizer
    )
    
    clear_environment()