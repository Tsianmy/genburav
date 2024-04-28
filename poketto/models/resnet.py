from torch import nn
from timm.models.resnet import ResNet, BasicBlock, Bottleneck

blocks = {'BasicBlock': BasicBlock, 'Bottleneck': Bottleneck}

class ResNet(ResNet):
    def __init__(self, **kwargs):
        fc_bias = kwargs.pop('fc_bias', True)
        block = kwargs['block']
        kwargs['block'] = blocks[block]
        super().__init__(**kwargs)
        if not fc_bias:
            self.fc = nn.Linear(self.num_features, self.num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data, mode='predict'):
        assert isinstance(data, dict)
        x = data['img']
        pred = super().forward(x)
        data['pred'] = pred
        if mode == 'loss':
            data['losses'] = self.loss(pred, data)

        return data
    
    def loss(self, pred, data):
        gt_label = data['gt_label']
        loss_ce = self.criterion(pred, gt_label)
        return dict(loss=loss_ce)