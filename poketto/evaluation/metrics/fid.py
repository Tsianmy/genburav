import torch
import torch.distributed as dist
import pytorch_fid.fid_score as fid
import numpy as np
from .base_metric import Metric

def get_activations(dataset, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- dataset     : Image tensors
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(dataset):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(dataset)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(dataset), dims))

    start_idx = 0

    for batch in dataloader:
        batch = batch.to(device)

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = fid.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_activation_statistics(dataset, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- dataset     : Image tensors
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(dataset, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

@torch.inference_mode()
def calculate_fid(fake_real, batch_size, device, dims, num_workers=1):
    block_idx = fid.InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = fid.InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(
        fake_real[0], model, batch_size, dims, device, num_workers
    )
    m2, s2 = calculate_activation_statistics(
        fake_real[1], model, batch_size, dims, device, num_workers
    )
    fid_value = fid.calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

class FrechetInceptionDistance(Metric):
    def __init__(self, batch_size=50, dims=2048):
        self.results = {'fake': [], 'real': []}
        self.metric_names = ['FID']
        self.valid = True
        self.batch_size = batch_size
        self.dims = dims
    
    def reset(self):
        for l in self.results.values():
            l.clear()
        self.valid = True

    def fetch(self, result):
        try:
            fake, real = result['pred'].detach(), result['img']
            if 'norm' in result:
                mean = torch.tensor(result['norm']['mean'], device=fake.device).view(-1, 1, 1)
                std = torch.tensor(result['norm']['std'], device=fake.device).view(-1, 1, 1)
                fake = fake * std + mean
                real = real * std + mean
                if not 'minmax' in result:
                    fake = fake / 255.
                    real = real / 255.
                fake = fake.clip(0, 1)
                real = real.clip(0, 1)
            self.results['fake'].append(fake)
            self.results['real'].append(real)
        except:
            for l in self.results.values():
                l.clear()
            self.valid = False

    def compute_metrics(self):
        fake = torch.cat(self.results['fake'])
        real = torch.cat(self.results['real'])
        vals = self.calculate(fake, real, batch_size=self.batch_size, dims=self.dims)
        metrics = {}
        for i, val in enumerate(vals):
            metrics[self.metric_names[i]] = val.item()
        for k in self.results:
            self.results[k].clear()
        return metrics
    
    @staticmethod
    def calculate(fake, real, **fid_args):
        if dist.is_initialized() and dist.get_world_size() > 1:
            raise NotImplementedError
        else:
            fid_args['batch_size'] = fid_args.get('batch_size', 50)
            fid_args['dims'] = fid_args.get('dims', 2048)
            fid_args['device'] = fake.device
            fid_score = calculate_fid([fake.cpu(), real.cpu()], **fid_args)
        return [fid_score]

    @staticmethod
    def get_best(fid_score):
        return min(fid_score)