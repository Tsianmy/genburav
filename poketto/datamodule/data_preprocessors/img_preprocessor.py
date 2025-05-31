import torch
from .base_preprocessor import BaseDataPreprocessor

class ImgDataPreprocessor(BaseDataPreprocessor):
    def __init__(
        self,
        *args,
        minmax=False,
        mean=None,
        std=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.minmax = minmax
        self.mean = mean
        self.std = std
        if mean is not None or std is not None:
            self._normalize = True
        else:
            self._normalize = False
    
    def __call__(self, data: dict, training=False):
        img = self.to_cuda(data['img'])

        img = img.float()
        if self.minmax:
            img.div_(255.)
            data['minmax'] = True
        if self._normalize:
            mean = torch.tensor(self.mean, device=self._device).view(-1, 1, 1)
            std = torch.tensor(self.std, device=self._device).view(-1, 1, 1)
            img.sub_(mean).div_(std)
            data['norm'] = dict(mean=self.mean, std=self.std)
        if training and self.batch_aug is not None:
            img = self.batch_aug(img)

        data['img'] = img
        return data
    
    def unnormalize_img(self, img):
        if self._normalize:
            mean = torch.tensor(self.mean, device=img.device).view(-1, 1, 1)
            std = torch.tensor(self.std, device=img.device).view(-1, 1, 1)
            img = img * std + mean
        if self.minmax:
            img = (img * 255)
        img = img.clip(0, 255)
        
        return img