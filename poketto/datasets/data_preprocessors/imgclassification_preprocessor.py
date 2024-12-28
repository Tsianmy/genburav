import torch
from torch.nn import functional as F
from .base import BaseDataPreprocessor

class ImgClsDataPreprocessor(BaseDataPreprocessor):
    def __init__(
        self,
        *args,
        minmax=False,
        mean=None,
        std=None,
        batch_aug=None,
        to_onehot=False,
        num_classes=None,
        label_smooth=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.minmax = minmax
        self.mean = mean
        self.std = std
        self.to_onehot = to_onehot
        self.num_classes = num_classes
        self.label_smooth = label_smooth
        if mean is not None:
            self._normalize = True
        else:
            self._normalize = False
        self.batch_aug = batch_aug
    
    def __call__(self, data: dict, training=False):
        img = self.to_cuda(data['img'])
        gt_label = self.to_cuda(data['gt_label'])

        img = img.float()
        if self.minmax:
            img.div_(255.)
        if self._normalize:
            mean = torch.tensor(self.mean, device=self._device).view(-1, 1, 1)
            std = torch.tensor(self.std, device=self._device).view(-1, 1, 1)
            img.sub_(mean).div_(std)
        if self.to_onehot:
            assert self.num_classes is not None
            gt_label = F.one_hot(gt_label, self.num_classes).float()

            if isinstance(self.label_smooth, float) and 0 <= self.label_smooth <= 1:
                off_value = self.label_smooth / self.num_classes
                on_value = 1. - self.label_smooth
                gt_label = gt_label * on_value + off_value
        if training and self.batch_aug is not None:
            img, gt_label = self.batch_aug(img, gt_label)

        data['img'] = img
        data['gt_label'] = gt_label
        return data
    