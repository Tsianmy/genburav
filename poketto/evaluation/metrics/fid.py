import torch
import torch.distributed as dist
from .base_metric import Metric
from .inception import InceptionV3

class FrechetInceptionDistance(Metric):
    def __init__(self, batch_size=0, dims=2048, use_amp=True):
        self.batch_size = batch_size
        self.dims = dims
        self.use_amp = use_amp
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.feature_net = InceptionV3([block_idx])
        self.feature_net.eval()
        self.metric_names = ['FID']
        self.reset()
    
    def reset(self):
        self.fake_sum = 0
        self.real_sum = 0
        self.fake_sigma_sum = 0
        self.real_sigma_sum = 0
        self.num_fake_samples = 0
        self.num_real_samples = 0

    @torch.inference_mode()
    def update(self, data) -> bool:
        try:
            fake, real = data['pred'].detach(), data['img'].detach()
        except:
            self.reset()
            return False
        if 'norm' in data:
            mean = torch.tensor(data['norm']['mean'], device=fake.device).view(-1, 1, 1)
            std = torch.tensor(data['norm']['std'], device=fake.device).view(-1, 1, 1)
            fake = fake * std + mean
            real = real * std + mean
            if not 'minmax' in data:
                fake = fake / 255.
                real = real / 255.
            fake = fake.clip(0, 1)
            real = real.clip(0, 1)
        fake_feat, real_feat = self.extract_feat(
            fake, real, self.feature_net, batch_size=self.batch_size, use_amp=self.use_amp
        )
        self.fake_sum += fake_feat.sum(0)
        self.fake_sigma_sum += torch.mm(fake_feat.t(), fake_feat)
        self.num_fake_samples += fake.shape[0]
        self.real_sum += real_feat.sum(0)
        self.real_sigma_sum += torch.mm(real_feat.t(), real_feat)
        self.num_real_samples += real.shape[0]
        return True

    @torch.inference_mode()
    def get_results(self):
        states = [
            self.fake_sum, self.fake_sigma_sum,
            torch.tensor([self.num_fake_samples], device=self.fake_sum.device),
            self.real_sum, self.real_sigma_sum,
            torch.tensor([self.num_real_samples], device=self.fake_sum.device)
        ]
        if dist.is_initialized() and dist.get_world_size() > 1:
            shapes = [tensor.shape for tensor in states]
            concat_states = torch.cat([tensor.view(-1) for tensor in states])
            dist.all_reduce(concat_states)
            offset = 0
            states = []
            for shape in shapes:
                numel = torch.prod(shape).item()
                states.append(concat_states[offset:offset+numel].view(shape))
                offset += numel
        (
            fake_sum, fake_sigma_sum, num_fake_samples,
            real_sum, real_sigma_sum, num_real_samples
        ) = states
        mu1, sigma1, mu2, sigma2 = self.calculate_statistics(
            fake_sum, fake_sigma_sum, num_fake_samples.item(),
            real_sum, real_sigma_sum, num_real_samples.item()
        )
        fid_score = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

        vals = [fid_score]
        metrics = {}
        for i, val in enumerate(vals):
            metrics[self.metric_names[i]] = val.item()

        return metrics
    
    @staticmethod
    @torch.inference_mode()
    def extract_feat(fake, real, model, **fid_args):
        batch_size = fid_args.get('batch_size', 0)
        if batch_size <= 0:
            batch_size = max(fake.shape[0], real.shape[0])
        use_amp = fid_args.get('use_amp', True)
        device = fake.device
        model.to(device)

        with torch.autocast(device.type, enabled=use_amp):
            fake_feat = model.extract_feat(fake, batch_size)
            real_feat = model.extract_feat(real, batch_size)
        
        return fake_feat.double(), real_feat.double()
    
    @staticmethod
    @torch.inference_mode()
    def calculate_statistics(
        fake_sum, fake_sigma_sum, num_fake_samples, real_sum, real_sigma_sum, num_real_samples
    ):
        m1, m2 = fake_sum / num_fake_samples, real_sum / num_real_samples
        s1 = fake_sigma_sum - num_fake_samples * torch.outer(m1, m1)
        s1 = s1 / (num_fake_samples - 1)
        s2 = real_sigma_sum - num_real_samples * torch.outer(m2, m2)
        s2 = s2 / (num_real_samples - 1)
        return m1, s1, m2, s2
    
    @staticmethod
    @torch.inference_mode()
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
        a = (mu1 - mu2).square().sum(dim=-1)
        b = sigma1.trace() + sigma2.trace()
        c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

        return a + b - 2 * c
    
    @staticmethod
    def get_best(fid_score):
        return min(fid_score)