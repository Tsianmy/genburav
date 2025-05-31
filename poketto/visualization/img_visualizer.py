import os
import matplotlib
import numpy as np
import torch
import torchvision
from .base_visualizer import BaseVisualizer

class ImgVisualizer(BaseVisualizer):
    def __init__(self, save_dir, use_tensorboard=False, tb_log_metrics=None, saving=False):
        super().__init__(save_dir, use_tensorboard, tb_log_metrics, saving)
        self.max_show = 8
    
    def add_data(self, data, dataset, step, **kwargs):
        if 'losses' in data:
            for k, v in data['losses'].items():
                self.add_scalar(f'{self.mode}/{k}', v, step)
        if 'metrics' in data:
            for k, v in data['metrics'].items():
                self.add_scalar(f'{self.mode}/{k}', v, step)
        if self.mode == 'eval' and 'last_batch' in data:
            results = data['last_batch']
            gt_img = results['img'][:self.max_show]
            pred = results['pred'][:self.max_show]
            if 'norm' in results:
                mean = torch.tensor(results['norm']['mean'], device=pred.device).view(-1, 1, 1)
                std = torch.tensor(results['norm']['std'], device=pred.device).view(-1, 1, 1)
                gt_img = gt_img * std + mean
                pred = pred * std + mean
            if 'minmax' in results:
                gt_img = gt_img * 255.
                pred = pred * 255.
            gt_img = gt_img.clip(0, 255).cpu()
            pred = pred.clip(0, 255).cpu()
            show_imgs = torch.cat([gt_img, pred]).to(torch.uint8)
            show_imgs = torchvision.utils.make_grid(show_imgs, nrow=4)
            self.add_image(f'{self.mode}/recons', show_imgs, step)
    
    def add_scalar(self, name, value, step):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(name, value, step)

    def add_image(self, name, image, step):
        if self.tb_writer is not None:
            self.tb_writer.add_image(name, image, step)

    def save(self, data, index=0):
        if not self.saving:
            return
        img = data['img']
        if 'norm' in data:
            mean = torch.tensor(data['norm']['mean'], device=img.device).view(-1, 1, 1)
            std = torch.tensor(data['norm']['std'], device=img.device).view(-1, 1, 1)
            img = img * std + mean
        if 'minmax' in data:
            img = img * 255.
        img = img.clip(0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        save_dir = os.path.join(self.save_dir, 'viz')
        os.makedirs(save_dir, exist_ok=True)
        for k, im in enumerate(img):
            matplotlib.pyplot.imsave(
                os.path.join(save_dir, f'{index + k:04d}.png'), im
            )