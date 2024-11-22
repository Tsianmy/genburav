import pickle
import os
import numpy as np
import torch
from torchvision.datasets.utils import check_integrity
from torchvision import tv_tensors
from .base import BaseDataset

class CIFAR10(BaseDataset):
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]
    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    def __init__(
        self,
        data_root='',
        transforms=None,
        test_mode=False,
        **kwargs
    ):
        super().__init__(
            data_root=data_root,
            transforms=transforms,
            test_mode=test_mode,
            **kwargs
        )
    
    def load_data_list(self):
        self.ann_file = os.path.join(self.data_root, self.base_folder, self.meta['filename'])
        if not check_integrity(self.ann_file, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(self.ann_file, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]

        if not self.test_mode:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        imgs = []
        gt_labels = []

        for file_name, _ in downloaded_list:
            file_path = os.path.join(self.data_root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                imgs.append(entry['data'])
                if 'labels' in entry:
                    gt_labels.extend(entry['labels'])
                else:
                    gt_labels.extend(entry['fine_labels'])

        imgs = np.vstack(imgs).reshape(-1, 3, 32, 32)
        imgs = torch.from_numpy(imgs)

        data_list = []
        for i, (img, gt_label) in enumerate(zip(imgs, gt_labels)):
            info = {'img': tv_tensors.Image(img),
                    'gt_label': int(gt_label),
                    'sample_idx': i,
                    'im_name': str(i)}
            data_list.append(info)
        return data_list
    
    def raw_data(self, idx):
        return self.data_list[idx]

class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }