from torchvision.transforms.v2 import RandomCrop
# import numpy as np
# import math
# import numbers
# from typing import Sequence, Optional, Union
# from poketto.ops.image import imcrop, impad

# class RandomCrop:
#     def __init__(self,
#                 crop_size: Union[Sequence, int],
#                 padding: Optional[Union[Sequence, int]] = None,
#                 pad_if_needed: bool = False,
#                 pad_val: Union[numbers.Number, Sequence[numbers.Number]] = 0,
#                 padding_mode: str = 'constant'):
#         if isinstance(crop_size, Sequence):
#             assert len(crop_size) == 2
#             assert crop_size[0] > 0 and crop_size[1] > 0
#             self.crop_size = crop_size
#         else:
#             assert crop_size > 0
#             self.crop_size = (crop_size, crop_size)
#         # check padding mode
#         assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
#         self.padding = padding
#         self.pad_if_needed = pad_if_needed
#         self.pad_val = pad_val
#         self.padding_mode = padding_mode

#     def rand_crop_params(self, img: np.ndarray):
#         """Get parameters for ``crop`` for a random crop.

#         Args:
#             img (ndarray): Image to be cropped.

#         Returns:
#             tuple: Params (offset_h, offset_w, target_h, target_w) to be
#                 passed to ``crop`` for random crop.
#         """
#         h, w = img.shape[:2]
#         target_h, target_w = self.crop_size
#         if w == target_w and h == target_h:
#             return 0, 0, h, w
#         elif w < target_w or h < target_h:
#             target_w = min(w, target_w)
#             target_h = min(w, target_h)

#         offset_h = np.random.randint(0, h - target_h + 1)
#         offset_w = np.random.randint(0, w - target_w + 1)

#         return offset_h, offset_w, target_h, target_w

#     def __call__(self, data: dict):
#         img = data['img']
#         if self.padding is not None:
#             img = impad(img, padding=self.padding, pad_val=self.pad_val)

#         if self.pad_if_needed:
#             h_pad = math.ceil(max(0, self.crop_size[0] - img.shape[0]) / 2)
#             w_pad = math.ceil(max(0, self.crop_size[1] - img.shape[1]) / 2)

#             img = impad(
#                 img,
#                 padding=(w_pad, h_pad, w_pad, h_pad),
#                 pad_val=self.pad_val,
#                 padding_mode=self.padding_mode)

#         offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
#         img = imcrop(
#             img,
#             np.array([
#                 offset_w,
#                 offset_h,
#                 offset_w + target_w - 1,
#                 offset_h + target_h - 1,
#             ]))
#         data['img'] = img
#         return data