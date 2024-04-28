from torchvision.transforms.v2 import RandomResizedCrop
# import numpy as np
# import math
# from typing import Union, Sequence, Tuple
# from poketto.ops.image import imcrop, imresize

# class RandomResizedCrop:
#     def __init__(self,
#                  scale: Union[Sequence, int],
#                  crop_ratio_range: Tuple[float, float] = (0.08, 1.0),
#                  aspect_ratio_range: Tuple[float, float] = (3. / 4., 4. / 3.),
#                  max_attempts: int = 10,
#                  interpolation: str = 'bilinear') -> None:
#         if isinstance(scale, Sequence):
#             assert len(scale) == 2
#             assert scale[0] > 0 and scale[1] > 0
#             self.scale = scale
#         else:
#             assert scale > 0
#             self.scale = (scale, scale)
#         if (crop_ratio_range[0] > crop_ratio_range[1]) or (
#                 aspect_ratio_range[0] > aspect_ratio_range[1]):
#             raise ValueError(
#                 'range should be of kind (min, max). '
#                 f'But received crop_ratio_range {crop_ratio_range} '
#                 f'and aspect_ratio_range {aspect_ratio_range}.')
#         assert isinstance(max_attempts, int) and max_attempts >= 0, \
#             'max_attempts mush be int and no less than 0.'
#         assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
#                                  'lanczos')

#         self.crop_ratio_range = crop_ratio_range
#         self.aspect_ratio_range = aspect_ratio_range
#         self.max_attempts = max_attempts
#         self.interpolation = interpolation

#     def rand_crop_params(self, img: np.ndarray) -> Tuple[int, int, int, int]:
#         """Get parameters for ``crop`` for a random sized crop.

#         Args:
#             img (ndarray): Image to be cropped.

#         Returns:
#             tuple: Params (offset_h, offset_w, target_h, target_w) to be
#                 passed to `crop` for a random sized crop.
#         """
#         h, w = img.shape[:2]
#         area = h * w

#         for _ in range(self.max_attempts):
#             target_area = np.random.uniform(*self.crop_ratio_range) * area
#             log_ratio = (math.log(self.aspect_ratio_range[0]),
#                          math.log(self.aspect_ratio_range[1]))
#             aspect_ratio = math.exp(np.random.uniform(*log_ratio))
#             target_w = int(round(math.sqrt(target_area * aspect_ratio)))
#             target_h = int(round(math.sqrt(target_area / aspect_ratio)))

#             if 0 < target_w <= w and 0 < target_h <= h:
#                 offset_h = np.random.randint(0, h - target_h + 1)
#                 offset_w = np.random.randint(0, w - target_w + 1)

#                 return offset_h, offset_w, target_h, target_w

#         # Fallback to central crop
#         in_ratio = float(w) / float(h)
#         if in_ratio < min(self.aspect_ratio_range):
#             target_w = w
#             target_h = int(round(target_w / min(self.aspect_ratio_range)))
#         elif in_ratio > max(self.aspect_ratio_range):
#             target_h = h
#             target_w = int(round(target_h * max(self.aspect_ratio_range)))
#         else:  # whole image
#             target_w = w
#             target_h = h
#         offset_h = (h - target_h) // 2
#         offset_w = (w - target_w) // 2
#         return offset_h, offset_w, target_h, target_w

#     def __call__(self, data: dict) -> dict:
#         """Transform function to randomly resized crop images.

#         Args:
#             data (dict): Result dict from loading pipeline.

#         Returns:
#             dict: Randomly resized cropped results
#         """
#         img = data['img']
#         offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
#         img = imcrop(
#             img,
#             bboxes=np.array([
#                 offset_w, offset_h, offset_w + target_w - 1,
#                 offset_h + target_h - 1
#             ]))
#         img = imresize(img, tuple(self.scale[::-1]),
#                        interpolation=self.interpolation)
#         data['img'] = img

#         return data