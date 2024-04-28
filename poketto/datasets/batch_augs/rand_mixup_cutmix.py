from torchvision.transforms.v2 import MixUp, CutMix, RandomChoice

class RandMixUpCutMix:
    def __init__(self, num_classes, switch_prob=0.5, alpha_mixup=1.0, alpha_cutmix=1.0):
        mixup = MixUp(alpha=alpha_mixup, num_classes=num_classes)
        cutmix = CutMix(alpha=alpha_cutmix, num_classes=num_classes)
        self._transform = RandomChoice([mixup, cutmix], p=[switch_prob, 1 - switch_prob])
    
    def __call__(self, *data):
        return self._transform(*data)