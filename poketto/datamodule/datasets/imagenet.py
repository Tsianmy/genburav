import os
from torchvision import io as tvio, tv_tensors
from .base_dataset import BaseDataset

class ImageNet(BaseDataset):
    """
    data_root/
        meta/
            train.txt
            val.txt
        train/
            n01440764/
                images
        val/
            n01440764/
                images
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
            self.ann_file = os.path.join(self.data_root, 'meta/val.txt')
            self.img_dir = os.path.join(self.data_root, 'val')
        else:
            self.ann_file = os.path.join(self.data_root, 'meta/train.txt')
            self.img_dir = os.path.join(self.data_root, 'train')
        with open(self.ann_file) as f:
            lines = f.readlines()
        samples = [x.strip().rsplit(' ', 1) for x in lines]

        data_list = []
        for sample in samples:
            filename, gt_label = sample
            img_path = os.path.join(self.img_dir, filename)
            info = {'img_path': img_path,
                    'gt_label': int(gt_label),
                    'im_name': filename}
            data_list.append(info)
        return data_list

    def raw_data(self, idx):
        data_info = self.data_list[idx]
        img = tvio.read_image(data_info['img_path'])
        data = {'img': tv_tensors.Image(img),
                'gt_label': data_info['gt_label'],
                'sample_idx': idx,
                'im_name': data_info['im_name']}

        return data