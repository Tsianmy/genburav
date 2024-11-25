import matplotlib.pyplot as plt
import numpy as np
from .base import BaseVisualizer
from poketto.utils import glogger

class VecDiffusionVisualizer(BaseVisualizer):
    def __init__(self, save_dir, use_tensorboard=False, tb_log_metrics=None):
        self.save_dir = save_dir
        self.mode = 'eval'
        self.tb_writer = None
        if use_tensorboard:
            glogger.info(f'[{self.__class__.__name__}] use Tensorboard')
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(self.save_dir)
                layout = {'losses': {'train': ['Multiline', ['train/loss*']]}}
                if tb_log_metrics is not None:
                    layout['metrics'] = {
                        k: ['Multiline', [f'^(train|eval)/{k}']] for k in tb_log_metrics}
                self.tb_writer.add_custom_scalars(layout)
            except:
                glogger.warning(f'[{self.__class__.__name__}] Tensorboard not available')
        self.save_data = {'vecs': []}
        self.add_gt_once = True

    def set_mode(self, name='eval'):
        assert name in ['train', 'eval']
        self.mode = name
    
    def fetch(self, data):
        if self.mode == 'eval':
            try:
                self.save_data['vecs'].append(data['pred'].cpu().numpy())
            except:
                pass

    def add_data(self, data, dataset, step):
        if 'losses' in data:
            for k, v in data['losses'].items():
                self.add_scalar(f'{self.mode}/{k}', v, step)
        if 'metrics' in data:
            for k, v in data['metrics'].items():
                self.add_scalar(f'{self.mode}/{k}', v, step)
        if self.mode == 'eval' and self.add_gt_once and 'last_batch' in data:
            gt_vecs = data['last_batch']['gt'].cpu().numpy()
            if gt_vecs.shape[1] == 2:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.scatter(gt_vecs[:, 0], gt_vecs[:, 1], color='red', edgecolor='white')
                ax.set_axis_off()
                ax.set_title(f'gt: $p(x)$', fontdict={'fontsize': 24})
                self.add_figure('eval/gt', fig, 0)
                self.add_gt_once = False
        if self.mode == 'eval' and len(self.save_data['vecs']) > 0:
            vecs = np.concatenate(self.save_data['vecs'], axis=0)
            if vecs.shape[1] == 2:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=20)
                ax.scatter(vecs[:, 0], vecs[:, 1], color='red', edgecolor='white')
                ax.set_axis_off()
                ax.set_title(
                    f'$p_\mathbf{{\\theta}}(x_0|x_1)$',
                    fontdict={'fontsize': 36}
                )
                self.add_figure(f'{self.mode}/predictions', fig, step)
            for l in self.save_data.values():
                l.clear()
    
    def add_scalar(self, name, value, step):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(name, value, step)

    def add_figure(self, name, image, step):
        if self.tb_writer is not None:
            self.tb_writer.add_figure(name, image, step)