from .cifar import CIFAR10, CIFAR100
from .imagenet import ImageNet
from .imagenetlmdb import ImageNetLMDB
from .tiny_imagenet import TinyImageNet
from .swissroll import SwissRoll
from .nerf_synth import NerfSynthetic

__all__ = [
    'CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNetLMDB', 'TinyImageNet',
    'SwissRoll', 'NerfSynthetic'
]