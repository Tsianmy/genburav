import torch
from torch import nn
from timm.models.resnet import ResNet as ResNet_, BasicBlock, Bottleneck, checkpoint_seq

blocks = {'BasicBlock': BasicBlock, 'Bottleneck': Bottleneck}

class ResNet_CIFAR(ResNet_):
    def __init__(self, **kwargs):
        block = kwargs['block']
        kwargs['block'] = blocks[block]
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2d(
            self.conv1.in_channels, self.conv1.out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.criterion = nn.CrossEntropyLoss()
        self.init_weights(zero_init_last=kwargs.get('zero_init_last', True))

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(
                [self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x

    def forward(self, data, inference_mode=False):
        assert isinstance(data, dict)
        x = data['img']
        pred = super().forward(x)
        data['pred'] = pred
        if self.training:
            data['losses'] = self.loss(pred, data)

        return data
    
    def loss(self, pred, data):
        gt_label = data['gt_label']
        loss_ce = self.criterion(pred, gt_label)
        return dict(loss=loss_ce)