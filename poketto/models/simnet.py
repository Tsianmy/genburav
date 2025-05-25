from torch import nn

class SimNet(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(4 * 4 * 128, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        self.act = nn.ReLU(inplace=True)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, data, inference_mode=False):
        assert isinstance(data, dict)
        x = data['img']
        pred = self._inner_forward(x)
        data['pred'] = pred
        if self.training:
            data['losses'] = self.loss(pred, data)

        return data
    
    def _inner_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.fc1(x.flatten(1))
        x = self.act(x)
        x = self.fc2(x)
        return x
    
    def loss(self, pred, data):
        gt_label = data['gt_label']
        loss_ce = self.criterion(pred, gt_label)
        return dict(loss=loss_ce)