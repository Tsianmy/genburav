from torchvision.transforms.v2 import RandomHorizontalFlip
# import numpy as np

# class RandomFlip:
#     valid_directions = ['horizontal', 'vertical', 'diagonal']
#     def __init__(self, prob, direction='horizontal'):
#         assert 0 <= prob <= 1
#         self.prob = prob

#         assert direction in self.valid_directions
#         self.direction = direction
    
#     def __call__(self, data: dict):
#         if np.random.random() < self.prob:
#             img = data['img']
#             if self.direction == 'horizontal':
#                 img =  np.flip(img, axis=1)
#             elif self.direction == 'vertical':
#                 img =  np.flip(img, axis=0)
#             else:
#                 img =  np.flip(img, axis=(0, 1))
            
#             data['img'] = img.copy()
#         return data