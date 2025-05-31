import os
import lmdb
import torch
import pickle as pkl
from torchvision import io as tvio, tv_tensors
from .base_dataset import BaseDataset

class ImageNetLMDB(BaseDataset):
    """
    data_root/
        train.lmdb
        val.lmdb
    """
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
        """Load image paths and gt_labels."""
        if self.test_mode:
            self.ann_file = os.path.join(self.data_root, 'val.lmdb')
        else:
            self.ann_file = os.path.join(self.data_root, 'train.lmdb')
        self.env = lmdb.open(
            self.ann_file, subdir=os.path.isdir(self.ann_file),
            readonly=True, lock=False,
            readahead=False, meminit=False
        )
        with self.env.begin(write=False) as txn:
            data_list = pkl.loads(txn.get(b'__keys__'))

        return data_list

    def raw_data(self, idx):
        key = self.data_list[idx]
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(key)
        unpacked = pkl.loads(byteflow)

        imgbuf = torch.frombuffer(unpacked[0], torch.uint8)
        img = tvio.decode_image(imgbuf)

        gt_label = unpacked[1]

        data = {'img': tv_tensors.Image(img),
                'gt_label': int(gt_label),
                'sample_idx': idx,
                'im_name': str(key)}

        return data