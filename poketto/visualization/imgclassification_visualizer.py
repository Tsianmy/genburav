import matplotlib.pyplot as plt
from .base_visualizer import BaseVisualizer
from .utils import matplotlib_imshow

class ImgClsVisualizer(BaseVisualizer):
    def add_data(self, data, dataset, step, **kwargs):
        if 'losses' in data:
            for k, v in data['losses'].items():
                self.add_scalar(f'{self.mode}/{k}', v, step)
        if 'metrics' in data:
            for k, v in data['metrics'].items():
                self.add_scalar(f'{self.mode}/{k}', v, step)
        if self.mode == 'eval' and 'last_batch' in data:
            results = data['last_batch']
            pred = results['pred'][-1:].cpu()
            pred = pred.softmax(1)
            sample_idx = results['sample_idx'][-1]
            im_name = results['im_name'][-1]
            raw_data = dataset.raw_data(sample_idx)
            raw_img = raw_data['img']
            raw_gt_label = raw_data['gt_label']
            fig = self.plot_classes_preds([raw_img], pred, [raw_gt_label], im_name)
            self.add_figure(f'{self.mode}/predictions', fig, step)
    
    def add_scalar(self, name, value, step):
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(name, value, step)

    def add_image(self, name, image, step):
        if self.tb_writer is not None:
            self.tb_writer.add_image(name, image, step)

    def add_figure(self, name, image, step):
        if self.tb_writer is not None:
            self.tb_writer.add_figure(name, image, step)

    def plot_classes_preds(self, images, probs, labels, im_name=None):
        n = 1
        fig = plt.figure(figsize=(8, 8 * n))
        preds = probs.argmax(1).tolist()
        probs = [p[i].item() for p, i in zip(probs, preds)]
        for idx in range(n):
            ax = fig.add_subplot(1, n, idx+1, xticks=[], yticks=[])
            matplotlib_imshow(images[idx])
            title = f'pred: {preds[idx]}, {probs[idx]*100:.1f}% (label: {labels[idx]})'
            if im_name is not None:
                title = f'{im_name}\n{title}'
            ax.set_title(
                title,
                fontdict={'fontsize': 24},
                color=("green" if preds[idx]==labels[idx] else "red")
            )
        return fig
    