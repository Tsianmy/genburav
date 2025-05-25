from torch import nn
from timm.models import resnet
from timm.models.resnet import BasicBlock, Bottleneck, \
    Tuple, Optional, LayerType, Type, Dict, Any

blocks = {'BasicBlock': BasicBlock, 'Bottleneck': Bottleneck}

class ResNet(resnet.ResNet):
    def __init__(
        self,
        block: str,
        layers: Tuple[int, ...],
        num_classes: int = 1000,
        in_chans: int = 3,
        output_stride: int = 32,
        global_pool: str = 'avg',
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = '',
        replace_stem_pool: bool = False,
        block_reduce_first: int = 1,
        down_kernel_size: int = 1,
        avg_down: bool = False,
        channels: Optional[Tuple[int, ...]] = (64, 128, 256, 512),
        act_layer: LayerType = nn.ReLU,
        norm_layer: LayerType = nn.BatchNorm2d,
        aa_layer: Optional[Type[nn.Module]] = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.,
        drop_block_rate: float = 0.,
        zero_init_last: bool = True,
        block_args: Optional[Dict[str, Any]] = None,
        fc_bias: bool = True,
        **kwargs
    ):
        block = blocks[block]
        super().__init__(
            block,
            layers,
            num_classes,
            in_chans,
            output_stride,
            global_pool,
            cardinality,
            base_width,
            stem_width,
            stem_type,
            replace_stem_pool,
            block_reduce_first, 
            down_kernel_size,
            avg_down, channels,
            act_layer,
            norm_layer,
            aa_layer,
            drop_rate,
            drop_path_rate,
            drop_block_rate,
            zero_init_last,
            block_args
        )
        if not fc_bias:
            self.fc = nn.Linear(self.num_features, self.num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss()

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