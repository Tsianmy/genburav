import os
from torchvision import io as tvio, tv_tensors
from .base import BaseDataset

class TinyImageNet(BaseDataset):
    """
    data_root/
        train/
            n01440764/
                images
        val/
            n01440764/
                images
    """
    def __init__(self,
                 data_root='',
                 transforms='',
                 test_mode=False,
                 **kwargs):

        super().__init__(
            data_root=data_root,
            transforms=transforms,
            test_mode=test_mode,
            **kwargs)
    
    def load_data_list(self):
        """Load image paths and gt_labels."""
        if self.test_mode:
            self.img_dir = os.path.join(self.data_root, 'val')
        else:
            self.img_dir = os.path.join(self.data_root, 'train')
        class_list = sorted(os.listdir(self.img_dir))

        data_list = []
        for gt_label in range(len(class_list)):
            cls_idx = class_list[gt_label]
            dirname = os.path.join(self.img_dir, cls_idx)
            for filename in sorted(os.listdir(dirname)):
                img_path = os.path.join(dirname, filename)
                info = {'img_path': img_path,
                        'gt_label': gt_label,
                        'im_name': filename}
                data_list.append(info)

        return data_list

    def prepare_data(self, idx):
        data_info = self.data_list[idx]
        img = tvio.read_image(data_info['img_path'])
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        data = {'img': tv_tensors.Image(img),
                'gt_label': data_info['gt_label'],
                'im_name': data_info['im_name']}

        return self.transforms(data)