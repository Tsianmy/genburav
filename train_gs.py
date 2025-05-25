import os
import sys
import time
import datetime
import argparse
import torch
from torch import distributed as dist
from omegaconf import OmegaConf
from poketto import factory
from poketto.utils import Tee, LossDict, RNGManager, create_logger
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--log_freq', type=int, default=2)
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
    config = OmegaConf.load(args.cfg)
    OmegaConf.set_struct(config, False)
    if args.override_cfg:
        config.merge_with_dotlist(args.override_cfg)
    if args.rank == 0:
        OmegaConf.save(config, os.path.join(args.output_dir, os.path.basename(args.cfg)))
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
    optimizer = state.get('optimizer')
    if optimizer is not None:
        state['optimizer'] = optimizer.state_dict()
    scheduler = state.get('scheduler')
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    grad_scaler = state.get('grad_scaler')
    if grad_scaler is not None:
        state['grad_scaler'] = grad_scaler.state_dict()
    rng = state.get('rng')
    if rng is not None:
        state['rng'] = rng.state_dict()
    torch.save(state, save_path)

def load_checkpoint(
    path_ckpt,
    config,
    model,
    optimizer,
    lr_scheduler=None,
    grad_scaler=None,
    rng_manager=None
):
    with open(path_ckpt, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu', weights_only=False)
    assert isinstance(checkpoint, dict), f'dict expected, but get {type(checkpoint)}'

    setattr(config, 'start_epoch', checkpoint['epoch'] + 1)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None and checkpoint.get('scheduler') is not None:
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    if grad_scaler is not None and checkpoint.get('grad_scaler') is not None:
        grad_scaler.load_state_dict(checkpoint['grad_scaler'])
    if rng_manager is not None and checkpoint.get('rng') is not None:
        rng_manager.load_state_dict(checkpoint['rng'])

def train(
    config,
    model,
    train_dataloader,
    val_dataloader,
    data_preprocessor,
    optimizer,
    evaluator,
    lr_scheduler=None,
    grad_scaler=None,
    rng_manager=None,
    visualizer=None
):
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.device_id], broadcast_buffers=False)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {num_params / 1e6}M")

    start_epoch = getattr(config, 'start_epoch', 0)
    ckpt_interval = getattr(config, 'ckpt_interval', 1)
    num_epochs = config.train_epochs
    if start_epoch > 0 and val_dataloader is not None:
        evaluate(
            config, model, val_dataloader, data_preprocessor, evaluator,
            start_epoch - 1, lr_scheduler, visualizer
        )
    iters_to_accumulate = getattr(config, 'iters_to_accumulate', 1)
    setattr(config, 'iters_to_accumulate', iters_to_accumulate)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        train_dataloader.sampler.set_epoch(epoch)

        train_one_epoch(
            config, model, train_dataloader, data_preprocessor,
            optimizer, epoch, evaluator, lr_scheduler, grad_scaler,
            visualizer
        )
        if config.rank == 0 and (epoch % ckpt_interval == 0 or epoch == num_epochs - 1):
            save_checkpoint(
                {'epoch': epoch, 'rng': rng_manager,
                 'model': model, 'optimizer': optimizer,
                 'scheduler': lr_scheduler, 'grad_scaler': grad_scaler},
                os.path.join(config.output_dir, 'latest.ckpt')
            )
        if val_dataloader is not None:
            evaluate(
                config, model, val_dataloader, data_preprocessor, evaluator,
                epoch, lr_scheduler, visualizer
            )

    total_time = time.time() - start_time
    logger.info(f'Training time {datetime.timedelta(seconds=int(total_time))}')

def train_one_epoch(
    config,
    model,
    dataloader,
    data_preprocessor,
    optimizer,
    epoch,
    evaluator,
    lr_scheduler=None,
    grad_scaler=None,
    visualizer=None
):
    model.train()
    evaluator.set_step(training=True, step=epoch, num_steps=config.train_epochs)
    if visualizer is not None:
        visualizer.set_mode('train')

    L = len(dataloader)
    log_interval = max(L // config.log_freq, 1)
    iters_to_accumulate = config.iters_to_accumulate
    max_grad_norm = 1.
    loss_dict = LossDict()
    start = time.time()
    data_end = time.time()
    end = time.time()
    for it, samples in enumerate(dataloader):
        samples = data_preprocessor(samples, training=True)
        data_time = time.time() - data_end
        
        global_it = it + epoch * L
        # model.module.update_learning_rate(global_it)
        # if (global_it + 1) % 1000 == 0:
        #     model.module.oneupSHdegree()

        require_backward_grad_sync = model.require_backward_grad_sync
        if (it + 1) % iters_to_accumulate != 0:
            model.require_backward_grad_sync = False

        with torch.amp.autocast('cuda', enabled=config.use_amp):
            results = model(samples)

        losses = results['losses']
        loss = losses['loss'] / config.iters_to_accumulate

        grad_norm = 0
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            # model.module.post_process(results, global_it)
            if (it + 1) % iters_to_accumulate == 0:
                grad_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                grad_scaler.step(optimizer, data=results, iteration=global_it)
                grad_scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            # model.module.post_process(results, global_it)
            if (it + 1) % iters_to_accumulate == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step(data=results, iteration=global_it)
                optimizer.zero_grad()
        model.require_backward_grad_sync = require_backward_grad_sync 
        
        batch_time = time.time() - end
        loss_dict.update(losses)
        evaluator.update(results)
        if visualizer is not None:
            visualizer.fetch(results)
        if it % log_interval == 0 or it == L - 1:
            if len(optimizer.param_groups) > 1:
                lr = ', '.join(f'{group["lr"]:.3g}' for group in optimizer.param_groups)
                lr = f'[{lr}]'
            else:
                lr = f'{optimizer.param_groups[0]["lr"]:.3g}'
            memory_used = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
            logger.info(
                f'Train [{epoch}][{it}/{L - 1}]  lr: {lr}  {loss_dict}  grad_norm: {grad_norm:.2f}'
                f'  time: {batch_time:.2f} (data {data_time:.2f})  mem: {memory_used:.0f}MB'
            )

        if lr_scheduler is not None:
            lr_scheduler.step_update(epoch * L + it + 1)
        
        end = time.time()
        data_end = time.time()

    metrics = evaluator.evaluate()
    if len(metrics) > 0:
        msg = ''
        for k, v in metrics.items():
            msg += f' {k}: {v:.3f}'
        logger.info(msg.strip())

    if visualizer is not None:
        vis_data = {
            'losses': {k: v.avg for k, v in loss_dict.avgs.items()},
            'metrics': {k: v for k, v in metrics.items()},
            'last_batch': results
        }
        visualizer.add_data(vis_data, dataloader.dataset, epoch)

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.inference_mode()
def evaluate(
    config,
    model,
    dataloader,
    data_preprocessor,
    evaluator,
    epoch,
    lr_scheduler=None,
    visualizer=None
):
    evaluator.set_step(training=False, step=epoch, num_steps=config.train_epochs)
    if not evaluator.ready():
        return
    model.eval()
    if visualizer is not None:
        visualizer.set_mode('eval')
    L = len(dataloader)
    log_interval = max(L // config.log_freq, 1)

    start = time.time()
    data_end = time.time()
    end = time.time()
    for it, samples in enumerate(dataloader):
        samples = data_preprocessor(samples, training=False)
        data_time = time.time() - data_end

        with torch.amp.autocast('cuda', enabled=config.use_amp):
            results = model(samples)

        evaluator.update(results)
        if visualizer is not None:
            visualizer.fetch(results)
            visualizer.save(results, it * dataloader.batch_size)
        
        batch_time = time.time() - end
        end = time.time()

        if it % log_interval == 0 or it == L - 1:
            memory_used = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
            logger.info(
                f'Val [{epoch}][{it}/{L - 1}]'
                f'  time: {batch_time:.2f} (data {data_time:.2f})  mem: {memory_used:.0f}MB'
            )
        
        data_end = time.time()
    
    metrics = evaluator.evaluate()
    
    msg = ' *'
    for k, v in metrics.items():
        msg += f' {k}: {v:.3f}'
    logger.info(msg)

    best_metrics = evaluator.best_eval_metrics
    msg = ' * best |'
    for k, v in best_metrics.items():
        msg += f' {k}: {v:.3f}'
    logger.info(msg)

    if lr_scheduler is not None and getattr(lr_scheduler, 'need_metric', False):
        lr_scheduler.metric = metrics[evaluator.primary_metric_key]

    if visualizer is not None:
        vis_data = {
            'metrics': {k: v for k, v in metrics.items()},
            'last_batch': results
        }
        visualizer.add_data(vis_data, dataloader.dataset, epoch)

    val_time = time.time() - start
    logger.info(f"evaluating takes {datetime.timedelta(seconds=int(val_time))}")

if __name__ == '__main__':
    args = parse_args()

    config, logger, rng_manager = setup_environment(args)

    logger.info(f'building dataloader')
    train_dataloader = factory.new_dataloader(
        config.train_dataloader, sampler_seed=rng_manager.seed)
    logger.info(f'train num: {len(train_dataloader.dataset)}')
    val_dataloader = None
    if getattr(config, 'val_dataloader', None) is not None:
        val_dataloader = factory.new_dataloader(config.val_dataloader)
        logger.info(f'val num: {len(val_dataloader.dataset)}')

    logger.info(f'building data_preprocessor {config.data_preprocessor["type"]}')
    data_preprocessor = factory.new_data_preprocessor(config.data_preprocessor)

    logger.info(f'building model {config.model["type"]}')
    model = factory.new_model(config.model, dataset=train_dataloader.dataset)
    logger.info(str(model))
    model.cuda()

    logger.info(f'building optimizer None')
    optimizer = factory.new_optimizer(config.optimizer, model)

    scheduler = None
    if getattr(config, 'scheduler', None) is not None:
        logger.info(f'building scheduler {config.scheduler["type"]}')
        scheduler = factory.new_scheduler(config.scheduler, optimizer, len(train_dataloader))

    logger.info(f'building evaluator')
    evaluator = factory.new_evaluator(config.evaluator)

    logger.info(f'building GradScaler')
    grad_scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)
    
    visualizer = None
    if config.rank == 0 and getattr(config, 'visualizer', None) is not None:
        logger.info(f'building visualizer {config.visualizer["type"]}')
        visualizer = factory.new_visualizer(
            config.visualizer, config.output_dir, tb_log_metrics=evaluator.metric_names)

    if config.resume:
        logger.info(f'loading checkpoint from {config.resume}')
        load_checkpoint(
            config.resume, config, model, optimizer,
            scheduler, grad_scaler, rng_manager
        )

    train(
        config,
        model,
        train_dataloader,
        val_dataloader,
        data_preprocessor,
        optimizer,
        evaluator,
        scheduler,
        grad_scaler,
        rng_manager,
        visualizer
    )

    clear_environment()