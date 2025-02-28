import time
import datetime
import argparse
import torch
from omegaconf import OmegaConf
from poketto import factory
from poketto.utils import create_logger, RNGManager

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, metavar="FILE", help='path to config file', )
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--log_freq', type=int, default=2)

    args, _ = parser.parse_known_args()

    return args

def setup_environment(args):
    logger = create_logger(0, name=__name__)

    ### load config
    config = OmegaConf.load(args.cfg)
    config.merge_with(args.__dict__)
    logger.info(f'Config\n{OmegaConf.to_yaml(config, resolve=True)}')

    ### set seed
    rng_manager = RNGManager()
    rng_manager.set_seed(config.seed)

    return config, logger

def load_model_checkpoint(path_ckpt, config, model):
    with open(path_ckpt, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu', weights_only=False)
    assert isinstance(checkpoint, dict), f'dict expected, but get {type(checkpoint)}'

    model.load_state_dict(checkpoint['model'])

@torch.no_grad()
def infer(config, model, dataloader, data_preprocessor, visualizer):
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"number of params: {num_params / 1e6}M")

    visualizer.set_mode('infer')
    logger.info("Start inference")
    start_time = time.time()

    model.eval()
    batch_time = 0
    L = len(dataloader)
    log_interval = max(L // config.log_freq, 1)

    end = time.time()
    for it, samples in enumerate(dataloader):
        samples = data_preprocessor(samples, training=False)
        with torch.amp.autocast('cuda', enabled=config.use_amp):
            results = model(samples, mode='inference')

        global_idx = it * dataloader.batch_size
        visualizer.save(results, index=global_idx)
        
        batch_time += time.time() - end
        end = time.time()

        if it % log_interval == 0:
            memory_used = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)
            logger.info(
                f'Infer [{it}/{L - 1}]  time: {batch_time:.2f}  mem: {memory_used:.0f}MB')

    total_time = time.time() - start_time
    logger.info(f'Inference time {datetime.timedelta(seconds=int(total_time))}')

if __name__ == '__main__':
    args = parse_args()

    config, logger = setup_environment(args)

    config.val_dataloader.pop('sampler')
    dataloader = factory.new_dataloader(config.val_dataloader)

    data_preprocessor = factory.new_data_preprocessor(config.data_preprocessor)

    model = factory.new_model(config.model)
    logger.info(str(model))
    model.cuda()

    if config.checkpoint:
        load_model_checkpoint(config.checkpoint, config, model)

    logger.info(f'building visualizer {config.visualizer["type"]}')
    config.visualizer.pop('use_tensorboard')
    visualizer = factory.new_visualizer(config.visualizer, config.output_dir)

    infer(
        config,
        model,
        dataloader,
        data_preprocessor,
        visualizer
    )
