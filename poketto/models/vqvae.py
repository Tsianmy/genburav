""" Neural Discrete Representation Learning
    - http://arxiv.org/abs/1711.00937
Code reference: https://github.com/airalcorn2/vqvae-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from poketto.utils import glogger

class ExponentialMovingAverage(nn.Module):
    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer('state', torch.zeros(shape, dtype=torch.float))
        self.register_buffer('average', torch.zeros(shape, dtype=torch.float))
    
    @torch.inference_mode()
    def update(self, value):
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(value)
        self.counter += 1
        self.state += (value - self.state) * (1 - self.decay)
        self.average = self.state / (1 - self.decay ** self.counter)
    
    def forward(self, value):
        self.update(value)
        return self.average

class VectorQuantizer(nn.Module):
    def __init__(self, num_embedding, embedding_dim, use_ema, decay=0.99, eps=1e-5, eini=-1):
        super().__init__()
        self.num_embedding = num_embedding
        self.use_ema = use_ema
        self.eps = eps
        if use_ema:
            limit = 3 ** 0.5
            embedding = torch.zeros(num_embedding, embedding_dim, dtype=torch.float).uniform_(
                -limit, limit
            )
            self.register_buffer('embedding', embedding)
            self.cluster_size_ema = ExponentialMovingAverage(decay, num_embedding)
            self.embedding_ema = ExponentialMovingAverage(
                decay, (num_embedding, embedding_dim)
            )
        else:
            if eini < 0:
                limit = embedding_dim ** -0.5 / 36
            elif eini > 0:
                limit = eini
            else:
                limit = 0
            embedding = torch.zeros(num_embedding, embedding_dim, dtype=torch.float).uniform_(
                -limit, limit
            )
            self.embedding = nn.Parameter(embedding)
        glogger.debug(f'[{self.__class__.__name__}] eini: {eini}, limit: {limit}')
    
    @torch.inference_mode()
    def update_ema(self, x, mapping_inds):
        ### (M, N)
        mapping_onehot = F.one_hot(mapping_inds, self.num_embedding).float().T
        cluster_size = mapping_onehot.sum(1)
        self.cluster_size_ema(cluster_size)
        
        cluster_sum = mapping_onehot @ x
        self.embedding_ema(cluster_sum) 
        
        cluster_size_avg_sum = self.cluster_size_ema.average.sum()
        cluster_size_stable = (
            (self.cluster_size_ema.average + self.eps)
            / (cluster_size_avg_sum + self.num_embedding * self.eps)
            * cluster_size_avg_sum
        )
        self.embedding = self.embedding_ema.average / cluster_size_stable[..., None]
    
    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            ### (N, M)
            dis = torch.cdist(
                x.detach()[None, ...], self.embedding.detach()[None, ...]).squeeze(0)
            mapping_inds = dis.argmin(dim=1)
            quantized_x = F.embedding(mapping_inds, self.embedding)
            loss_commitment = (x - quantized_x.detach()).square().mean()
            losses = dict(loss_commitment=loss_commitment)
            if not self.use_ema:
                loss_dictionary = (x.detach() - quantized_x).square().mean()
                losses['loss_dictionary'] = loss_dictionary
            ### Straight-through gradient
            quantized_x = x + (quantized_x - x).detach()
            
            if self.use_ema and self.training:
                self.update_ema(x, mapping_inds)

        return (quantized_x, losses, mapping_inds)

class ResidualBlock(nn.Module):
    def __init__(self, chs, hidden_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(chs, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, chs, kernel_size=1)
        )
    
    def forward(self, x):
        return x + self.conv(x)

class Decoder(nn.Module):
    def __init__(
        self,
        out_channels,
        up_hidden_dim,
        res_hidden_dim,
        num_upsampling,
        num_res_block,
        embedding_dim
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(embedding_dim, up_hidden_dim, kernel_size=3, padding=1)
        conv2 = [ResidualBlock(up_hidden_dim, res_hidden_dim) for _ in range(num_res_block)]
        self.conv2 = nn.Sequential(*conv2, nn.ReLU())
        conv3 = []
        out_chs = out_channels
        for i in range(num_upsampling):
            if i == 0:
                in_chs = up_hidden_dim // 2
            else:
                in_chs = up_hidden_dim
            if i > 0:
                conv3.append(nn.ReLU())
            conv3.append(
                nn.ConvTranspose2d(in_chs, out_chs, kernel_size=4, stride=2, padding=1)
            )
            out_chs = in_chs
        self.conv3 = nn.Sequential(*conv3[::-1])
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        down_hidden_dim,
        res_hidden_dim,
        num_downsampling,
        num_res_block,
        embedding_dim
    ):
        super().__init__()
        conv1 = []
        in_chs = in_channels
        for i in range(num_downsampling):
            if i == 0:
                out_chs = down_hidden_dim // 2
            else:
                out_chs = down_hidden_dim
            conv1.append(nn.Conv2d(in_chs, out_chs, kernel_size=4, stride=2, padding=1))
            conv1.append(nn.ReLU())
            in_chs = out_chs
        conv1.append(nn.Conv2d(in_chs, down_hidden_dim, kernel_size=3, stride=1, padding=1))
        self.conv1 = nn.Sequential(*conv1)
        conv2 = [ResidualBlock(down_hidden_dim, res_hidden_dim) for _ in range(num_res_block)]
        self.conv2 = nn.Sequential(*conv2, nn.ReLU())
        self.conv3 = nn.Conv2d(down_hidden_dim, embedding_dim, kernel_size=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        sampling_hidden_dim,
        res_hidden_dim,
        num_sampling,
        num_res_block,
        embedding_dim,
        num_embedding,
        use_ema=True,
        alpha=1.0,
        beta=0.25,
        ema_decay=0.99,
        ema_epsilon=1e-5,
        eini=-1
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.encoder = Encoder(
            in_channels, sampling_hidden_dim, res_hidden_dim,
            num_sampling, num_res_block, embedding_dim
        )
        self.vq = VectorQuantizer(
            num_embedding, embedding_dim, use_ema, ema_decay, ema_epsilon, eini
        )
        self.decoder = Decoder(
            in_channels, sampling_hidden_dim, res_hidden_dim,
            num_sampling, num_res_block, embedding_dim
        )
    
    def forward(self, data, inference_mode=False):
        x = data['img']
        x = self.encoder(x)
        B, C, H ,W = x.shape
        x = x.permute(0, 2, 3, 1).flatten(0, 2)
        outs = self.vq(x)
        x, precomp_losses, mapping_inds = outs 
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.decoder(x)
        data['pred'] = x
        if self.training:
            data['losses'] = self.loss(x, data, **precomp_losses)
        
        return data
    
    def loss(self, pred, data, **precomp_losses):
        gt = data['img']
        loss_mse = (gt - pred).square().mean() * self.alpha
        precomp_losses['loss_commitment'] *= self.beta
        total_loss = loss_mse
        for loss in precomp_losses.values():
            total_loss = total_loss + loss

        return dict(
            loss=total_loss,
            loss_recon_mse=loss_mse,
            **precomp_losses
        )

if __name__ == '__main__':
    model = VQVAE(
        in_channels=3, sampling_hidden_dim=128, res_hidden_dim=32,
        num_sampling=2, num_res_block=2, embedding_dim=64, num_embedding=512,
        use_ema=True
    )
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'num_params: {num_params / 1e6} M')
    data = {'img': torch.rand(2, 3, 32, 32)}
    data = model(data)
    print(f"prediction: {data['pred'].shape}")
    print(f"losses: {data['losses']}")
