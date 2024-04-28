import os
import sys
import time
import datetime
import argparse
import torch
from torch import distributed as dist
from omegaconf import OmegaConf
from poketto.build import (build_model, build_dataloader, build_optimizer,
                         build_scheduler, build_evaluator, build_data_preprocessor)
from poketto.utils import Tee, LossDict, RNGManager, create_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--resume', type=str)
    parser.add_argument('override_cfg', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    args.rank = int(os.environ['RANK'])
    args.device_id = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])

    return args

def setup_environment(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logfile = os.path.join(args.output_dir, 'log.txt')
    tee = Tee(sys.stdout, logfile, mode='a')
    sys.stdout, sys.stderr = tee, tee
    logger = create_logger(args.rank, name=__name__)
    if args.rank == 0:
        print(' '.join(sys.argv))

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.device_id)
    print(f"RANK and WORLD_SIZE: {args.rank}/{args.world_size}")

    ### load config
    config = None
    if args.rank == 0:
        config = OmegaConf.load(args.cfg)
        OmegaConf.set_struct(config, False)
        if args.override_cfg:
            config.merge_with_dotlist(args.override_cfg)
        OmegaConf.save(config, os.path.join(args.output_dir, os.path.basename(args.cfg)))
    sync_objs = [config]
    dist.broadcast_object_list(sync_objs, src=0)
    config = sync_objs[0]

    config.merge_with({k: v for k, v in args.__dict__.items() if k != 'override_cfg'})
    logger.info(f'Config\n{OmegaConf.to_yaml(config, resolve=True)}')

    ### set seed
    rng_manager = RNGManager()
    rng_manager.set_seed(config.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic  = True

    return config, logger, rng_manager

def clear_environment():
    dist.destroy_process_group()

def save_checkpoint(state, save_path):
    print(f'save checkpoint to {save_path}')
    model = state['model']
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module.state_dict()
    elif isinstance(model, torch.nn.Module):
        model = model.state_dict()
    state['model'] = model
    optimizer = state['optimizer'].state_dict()
    state['optimizer'] = optimizer
    scheduler = state.get('scheduler', None)
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    grad_scaler = state.get('grad_scaler', None)
    if grad_scaler is not None:
        state['grad_scaler'] = grad_scaler.state_dict()
    rng = state.get('rng', None)
    if rng is not None:
        state['rng'] = rng.state_dict()
    torch.save(state, save_path)

def load_checkpoint(path_ckpt, config, model, optimizer,
                    lr_scheduler=None, grad_scaler=None, rng_manager=None):
    with open(path_ckpt, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')
    assert isinstance(checkpoint, dict), f'dict expected, but get {type(checkpoint)}'

    setattr(config, 'start_epoch', checkpoint['epoch'] + 1)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None and checkpoint.get('scheduler', None) is not None:
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    if grad_scaler is not None and checkpoint.get('grad_scaler', None) is not None:
        grad_scaler.load_state_dict(checkpoint['grad_scaler'])
    if rng_manager is not None and checkpoint.get('rng', None) is not None:
        rng_manager.load_state_dict(checkpoint['rng'])

def train(config, model, train_dataloader, val_dataloader, data_preprocessor, optimizer,
          evaluator, lr_scheduler=None, grad_scaler=None, rng_manager=None):
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.device_id], broadcast_buffers=False)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {num_params / 1e6}M")

    start_epoch = getattr(config, 'start_epoch', 0)
    if start_epoch > 0:
        evaluate(config, model, val_dataloader, data_preprocessor, evaluator,
                 start_epoch - 1, lr_scheduler)
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config.train_epochs):
        train_dataloader.sampler.set_epoch(epoch)

        train_one_epoch(config, model, train_dataloader, data_preprocessor,
                        optimizer, epoch, lr_scheduler, grad_scaler)

        if config.rank == 0:
            save_checkpoint(
                {'epoch': epoch, 'rng': rng_manager,
                 'model': model, 'optimizer': optimizer,
                 'scheduler': lr_scheduler, 'grad_scaler': grad_scaler},
                os.path.join(config.output_dir, 'latest.ckpt'))

        evaluate(config, model, val_dataloader, data_preprocessor, evaluator,
                 epoch, lr_scheduler)


    total_time = time.time() - start_time
    logger.info(f'Training time {datetime.timedelta(seconds=int(total_time))}')

def train_one_epoch(config, model, dataloader, data_preprocessor, optimizer,
                    epoch, lr_scheduler=None, grad_scaler=None):
    model.train()

    L = len(dataloader)
    loss_dict = LossDict()
    start = time.time()
    data_end = time.time()
    end = time.time()
    for it, samples in enumerate(dataloader):
        samples = data_preprocessor(samples, training=True)
        data_time = time.time() - data_end
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.use_amp):
            results = model(samples, mode='loss')

        losses = results['losses']
        loss = losses['loss']

        grad_norm = 0
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        batch_time = time.time() - end
        loss_dict.update(losses)
        if it % config.log_interval == 0 or it == L - 1:
            if len(optimizer.param_groups) > 1:
                lr = [f'{group["lr"]:.3g}' for group in optimizer.param_groups]
            else:
                lr = f'{optimizer.param_groups[0]["lr"]:.3g}'
            memory_used = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
            logger.info(
                f'Train [{epoch}][{it}/{L - 1}]  lr: {lr}  {loss_dict}  grad_norm: {grad_norm:.2f}'
                f'  time: {batch_time:.2f} (data {data_time:.2f})  mem: {memory_used:.0f}MB'
            )
        end = time.time()

        if lr_scheduler is not None:
            lr_scheduler.step_update(epoch * L + it + 1)
        
        data_end = time.time()

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def evaluate(config, model, dataloader, data_preprocessor, evaluator, epoch,
             lr_scheduler=None):
    model.eval()
    loss_dict = LossDict()
    L = len(dataloader)

    start = time.time()
    data_end = time.time()
    end = time.time()
    for it, samples in enumerate(dataloader):
        samples = data_preprocessor(samples, training=False)
        data_time = time.time() - data_end

        with torch.cuda.amp.autocast(enabled=config.use_amp):
            results = model(samples, mode='loss')

        losses = results.pop('losses')
        loss_dict.update(losses)

        evaluator.fetch(results)
        
        batch_time = time.time() - end
        end = time.time()

        if it % config.log_interval == 0 or it == L - 1:
            memory_used = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
            logger.info(f'Val [{epoch}][{it}/{L - 1}]  {loss_dict}'
                f'  time: {batch_time:.2f} (data {data_time:.2f})  mem: {memory_used:.0f}MB')
        
        data_end = time.time()
    
    metrics = evaluator.evaluate()
    
    msg = ' *'
    for k, v in metrics.items():
        msg += f' {k}: {v:.3f}'
    logger.info(msg)

    best_metrics = evaluator.best_eval_metrics
    msg = 'best |'
    for k, v in best_metrics.items():
        msg += f' {k}: {v:.3f}'
    logger.info(msg)

    if lr_scheduler is not None and getattr(lr_scheduler, 'need_metric', False):
        lr_scheduler.metric = metrics[evaluator.primary_metric_key]

    val_time = time.time() - start
    logger.info(f"evaluating takes {datetime.timedelta(seconds=int(val_time))}")

    return metrics, best_metrics

if __name__ == '__main__':
    args = parse_args()

    config, logger, rng_manager = setup_environment(args)

    logger.info(f'building dataloader')
    train_dataloader = build_dataloader(config.train_dataloader,
                                        sampler_seed=rng_manager.seed)
    val_dataloader = build_dataloader(config.val_dataloader)
    logger.info(f'train num: {len(train_dataloader.dataset)}')
    logger.info(f'val num: {len(val_dataloader.dataset)}')

    logger.info(f'building data_preprocessor {config.data_preprocessor["type"]}')
    data_preprocessor = build_data_preprocessor(config.data_preprocessor)

    logger.info(f'building model {config.model["type"]}')
    model = build_model(config.model)
    logger.info(str(model))
    model.cuda()

    logger.info(f'building optimizer {config.optimizer["type"]}')
    optimizer = build_optimizer(config.optimizer, model.parameters())

    scheduler = None
    if getattr(config, 'scheduler', None) is not None:
        logger.info(f'building scheduler {config.scheduler["type"]}')
        scheduler = build_scheduler(config.scheduler, optimizer, len(train_dataloader))

    logger.info(f'building evaluator')
    evaluator = build_evaluator(config.evaluator)

    logger.info(f'building GradScaler')
    grad_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=config.use_amp)

    if config.resume:
        logger.info(f'loading checkpoint from {config.resume}')
        load_checkpoint(config.resume, config, model, optimizer,
                        scheduler, grad_scaler, rng_manager)

    train(config, model, train_dataloader, val_dataloader, data_preprocessor,
          optimizer, evaluator, scheduler, grad_scaler, rng_manager)

    clear_environment()