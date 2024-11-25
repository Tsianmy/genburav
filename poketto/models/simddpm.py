""" Denoising Diffusion Probabilistic Models
    - https://arxiv.org/abs/2006.11239
Code reference: https://github.com/StarLight1212/Generative-models/tree/main/Diffusion-Model
@Author: Alex
@Date: 2022.Dec.6th
Diffusion model
"""
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, in_chs, num_steps, num_units):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(in_chs, num_units),
            nn.Linear(num_units, num_units),
            nn.Linear(num_units, num_units),
            nn.Linear(num_units, in_chs),
        ])
        self.act = nn.ReLU()
        self.step_embeddings = nn.ModuleList(
            [nn.Embedding(num_steps, num_units) for _ in range(len(self.linears) - 1)])
    
    def forward(self, x, t):
        for i in range(len(self.step_embeddings)):
            x = self.linears[i](x)
            t_embedding = self.step_embeddings[i](t)
            x += t_embedding
            x = self.act(x)
        x = self.linears[-1](x)
        return x

class SimDDPM(nn.Module):
    def __init__(self, in_chs, num_steps, num_units=128):
        super().__init__()
        self.num_steps = num_steps
        ### hyperparameters
        betas = torch.linspace(-6, 6, num_steps)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, dim=0)
        one_minus_alphas_prod = 1 - alphas_prod
        self.betas = nn.Parameter(betas, requires_grad=False)
        self.betas_sqrt = nn.Parameter(betas.sqrt(), requires_grad=False)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.alphas_sqrt = nn.Parameter(alphas.sqrt(), requires_grad=False)
        self.alphas_prod_sqrt = nn.Parameter(alphas_prod.sqrt(), requires_grad=False)
        self.one_minus_alphas_prod_sqrt = nn.Parameter(
            one_minus_alphas_prod.sqrt(), requires_grad=False)
        ### decoder
        self.decoder = Decoder(in_chs, num_steps, num_units)
    
    def forward(self, data, mode='predict'):
        x0 = data['gt']
        noise = torch.randn(x0.shape, device=x0.device)
        if mode == 'loss':
            B = x0.shape[0]
            t = torch.randint(0, self.num_steps, size=(B // 2,), device=x0.device)
            t = torch.cat([t, self.num_steps - 1 - t], dim=0)
            x = self.alphas_prod_sqrt[t].unsqueeze(-1) * x0 + \
                    self.one_minus_alphas_prod_sqrt[t].unsqueeze(-1) * noise
            pred_noise = self.decoder(x, t)
            data['losses'] = self.loss(pred_noise, noise)
        if mode == 'predict':
            with torch.no_grad():
                xt = noise
                for t in reversed(range(self.num_steps)):
                    pred_noise = self.decoder(xt, torch.tensor([t], device=xt.device))
                    noise_coeff = self.betas[t] / self.one_minus_alphas_prod_sqrt[t]
                    mean = (xt - noise_coeff * pred_noise) / self.alphas_sqrt[t]
                    sigma_t = self.betas_sqrt[t]
                    z = torch.randn_like(xt)
                    xt = mean + sigma_t * z
                data['pred'] = xt
        return data
    
    def loss(self, pred_noise, gt_noise):
        loss_mse = (gt_noise - pred_noise).square().mean()
        return dict(loss=loss_mse)