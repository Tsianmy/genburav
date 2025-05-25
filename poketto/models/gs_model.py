""" 3D Gaussian Splatting for Real-Time Radiance Field Rendering
    - http://arxiv.org/abs/2308.04079
Code reference: https://github.com/graphdeco-inria/gaussian-splatting

Copyright (C) 2023, Inria
GRAPHDECO research group, https://team.inria.fr/graphdeco
All rights reserved.

This software is free for non-commercial, research and evaluation use 
under the terms of the LICENSE.md file.

For inquiries contact  george.drettakis@inria.fr

"""
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from poketto.ops.sh_utils import RGB2SH, eval_sh
from poketto.losses import ssim
from poketto.utils import glogger

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = q.new_zeros((q.size(0), 3, 3))

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = s.new_zeros((s.shape[0], 3, 3), dtype=torch.float)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def strip_lowerdiag(L):
    uncertainty = L.new_zeros((L.shape[0], 6), dtype=torch.float)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class GaussianSplatting(nn.Module):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(
        self,
        sh_degree,
        dataset,
        percent_dense=0.01,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_grad_threshold=0.0002,
        lambda_dssim=0.2,
        **kwargs
    ):
        super().__init__()
        self.setup_functions()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  

        self.cameras_extent = dataset.nerf_normalization['radius']
        self.create_from_pcd(dataset.pcd)
        self.white_background = dataset.white_background

        self.register_buffer('xyz_gradient_accum', torch.zeros((self.get_xyz.shape[0], 1)))
        self.register_buffer('denom', torch.zeros((self.get_xyz.shape[0], 1)))

        self.percent_dense = percent_dense
        self.densification_interval = densification_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densify_grad_threshold = densify_grad_threshold

        self.l1_loss = nn.L1Loss()
        self.ssim = ssim
        self.lambda_dssim = lambda_dssim

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd):
        fused_point_cloud = pcd.points.float()
        fused_color = RGB2SH(pcd.colors.float())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float()#.cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        glogger.info(f'Number of points at initialisation: {fused_point_cloud.shape[0]}')

        dist2 = torch.clamp_min(distCUDA2(pcd.points.float().cuda()).cpu(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.register_buffer('max_radii2D', torch.zeros((self.get_xyz.shape[0])))

    def reset_opacity(self, param_operations):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        self._opacity = nn.Parameter(opacities_new.requires_grad_(True))

        param_operations.append({
            'type': 'replace',
            'shape': {
                'opacity': self._opacity.shape,
            }
        })

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self):
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        return PlyData([el])

    def load_ply(self, plydata):
        raise NotImplementedError
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        device = self.get_xyz.device
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
    
    def state_dict(self, *args, **kwargs):
        plydata = self.save_ply()
        return {'plydata': plydata}
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        plydata = state_dict['plydata']
        self.load_ply(plydata)
        return ([], [])

    def prune_points(self, param_operations, mask):
        valid_points_mask = ~mask

        self._xyz = nn.Parameter(self._xyz[valid_points_mask].contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(self._features_dc[valid_points_mask].contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(self._features_rest[valid_points_mask].contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(self._opacity[valid_points_mask].contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(self._scaling[valid_points_mask].contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(self._rotation[valid_points_mask].contiguous().requires_grad_(True))

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask].contiguous()

        self.denom = self.denom[valid_points_mask].contiguous()
        self.max_radii2D = self.max_radii2D[valid_points_mask].contiguous()

        param_operations.append({
            'type': 'prune',
            'params': ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation'],
            'mask': valid_points_mask
        })

    def densification_postfix(
        self, param_operations, new_xyz, new_features_dc, new_features_rest,
        new_opacities, new_scaling, new_rotation
    ):
        self._xyz = nn.Parameter(torch.cat((self._xyz, new_xyz)).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.cat((self._features_dc, new_features_dc)).requires_grad_(True))
        self._features_rest = nn.Parameter(torch.cat((self._features_rest, new_features_rest)).requires_grad_(True))
        self._opacity = nn.Parameter(torch.cat((self._opacity, new_opacities)).requires_grad_(True))
        self._scaling = nn.Parameter(torch.cat((self._scaling, new_scaling)).requires_grad_(True))
        self._rotation = nn.Parameter(torch.cat((self._rotation, new_rotation)).requires_grad_(True))

        self.xyz_gradient_accum = self.xyz_gradient_accum.new_zeros((self.get_xyz.shape[0], 1))
        self.denom = self.denom.new_zeros((self.get_xyz.shape[0], 1))
        self.max_radii2D = self.max_radii2D.new_zeros(self.get_xyz.shape[0])

        param_operations.append({
            'type': 'add',
            'shape': {
                'xyz': new_xyz.shape,
                'f_dc': new_features_dc.shape,
                'f_rest': new_features_rest.shape,
                'opacity': new_opacities.shape,
                'scaling': new_scaling.shape,
                'rotation': new_rotation.shape
            }
        })

    def densify_and_split(self, param_operations, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        device = self.get_xyz.device
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(param_operations, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=device, dtype=bool)))
        self.prune_points(param_operations, prune_filter)

    def densify_and_clone(self, param_operations, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(param_operations, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, param_operations, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(param_operations, grads, max_grad, extent)
        self.densify_and_split(param_operations, grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(param_operations, prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        valid_j = update_filter.any(dim=0)
        grad_norms = torch.linalg.norm(viewspace_point_tensor.grad[..., :2], dim=-1)
        summed_grads = (grad_norms * update_filter).sum(dim=0, keepdim=True).T
        update_counts = update_filter.sum(dim=0, keepdim=True).T
        self.xyz_gradient_accum[valid_j] += summed_grads[valid_j]
        self.denom[valid_j] += update_counts[valid_j]

    def forward(self, data, inference_mode=False):
        batch_render = []
        batch_visibility_filter = []
        batch_radii = []
        tanfovx = torch.tan(data['FovX'] * 0.5)
        tanfovy = torch.tan(data['FovY'] * 0.5)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        batch_screen_points = self.get_xyz.new_zeros(
            (tanfovx.shape[0], *self.get_xyz.shape), requires_grad=True
        ) + 0
        if self.training:
            batch_screen_points.retain_grad()
        for i in range(tanfovx.shape[0]):
            device = self.get_xyz.device

            # Set up rasterization configuration
            
            bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
            bg_color = torch.tensor(bg_color, dtype=torch.float, device=device)

            raster_settings = GaussianRasterizationSettings(
                image_height=int(data['height'][i]),
                image_width=int(data['width'][i]),
                tanfovx=tanfovx[i].item(),
                tanfovy=tanfovy[i].item(),
                bg=bg_color,
                scale_modifier=1.0,
                viewmatrix=data['world_view_transform'][i],
                projmatrix=data['full_proj_transform'][i],
                sh_degree=self.active_sh_degree,
                campos=data['camera_center'][i],
                prefiltered=False,
                debug=False,
                antialiasing=False
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            means3D = self.get_xyz
            # means2D = screenspace_points
            means2D = batch_screen_points[i]
            opacity = self.get_opacity

            # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
            # scaling / rotation by the rasterizer.
            scales = None
            rotations = None
            cov3D_precomp = None
            compute_cov3D_python = False
            if compute_cov3D_python:
                cov3D_precomp = self.get_covariance(1.0)
            else:
                scales = self.get_scaling
                rotations = self.get_rotation

            # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
            # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
            shs = None
            colors_precomp = None
            convert_SHs_python = False
            override_color = None
            if override_color is None:
                if convert_SHs_python:
                    shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
                    dir_pp = (self.get_xyz - data['camera_center'][i].repeat(self.get_features.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                else:
                    shs = self.get_features
            else:
                colors_precomp = override_color

            # Rasterize visible Gaussians to image, obtain their radii (on screen). 
            rendered_image, radii, rendered_depth = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
            
            # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
            # They will be excluded from value updates used in the splitting criteria.
            batch_render.append(rendered_image)
            batch_visibility_filter.append(radii > 0)
            batch_radii.append(radii)
        
        data['pred'] = torch.stack(batch_render)
        data['viewspace_points'] = batch_screen_points
        data['visibility_filter'] = torch.stack(batch_visibility_filter)
        data['radii'] = torch.stack(batch_radii)

        if self.training:
            gt_image = data['img']
            batch_render = data['pred']
            Ll1 = self.l1_loss(batch_render, gt_image)
            loss_ssim = 1.0 - self.ssim(batch_render, gt_image)
            loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * loss_ssim
            losses = dict(loss=loss, loss_l1=Ll1, loss_ssim=loss_ssim)
            data['losses'] = losses

        return data
    
    def optimizer_step_pre_hook(self, optimizer, args, kwargs):
        param_operations = []
        iteration = kwargs.pop('config').global_iter
        if iteration < self.densify_until_iter:
            data = kwargs.pop('data')
            visibility_filter = data['visibility_filter']
            radii = data['radii']
            viewspace_point_tensor = data['viewspace_points']
            # Keep track of max radii in image-space for pruning
            valid_j = visibility_filter.any(dim=0)
            masked_radii = torch.where(visibility_filter, radii, -torch.inf)
            max_radii_per_j = masked_radii.max(dim=0).values
            self.max_radii2D[valid_j] = torch.maximum(self.max_radii2D[valid_j], max_radii_per_j[valid_j])
            self.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration >= self.densify_from_iter and (iteration + 1) % self.densification_interval == 0:
                size_threshold = 20 if iteration > self.opacity_reset_interval else None
                self.densify_and_prune(param_operations, self.densify_grad_threshold, 0.005, self.cameras_extent, size_threshold)
            
            if (iteration + 1) % self.opacity_reset_interval == 0 or (self.white_background and (iteration + 1) == self.densify_from_iter):
                self.reset_opacity(param_operations)
        if (iteration + 1) % 1000 == 0:
            self.oneupSHdegree()
        if len(param_operations) > 0:
            params = {
                'xyz': self._xyz,
                'f_dc': self._features_dc,
                'f_rest': self._features_rest,
                'opacity': self._opacity,
                'scaling': self._scaling,
                'rotation': self._rotation
            }
            optimizer.sync_model_params(params, param_operations)

    def extra_repr(self) -> str:
        _repr = (
            f'xyz: {self._xyz.shape}\n'
            f'f_dc: {self._features_dc.shape}\n'
            f'f_rest: {self._features_rest.shape}\n'
            f'opacity: {self._opacity.shape}\n'
            f'scaling: {self._scaling.shape}\n'
            f'rotation: {self._rotation.shape}\n'
            f'percent_dense: {self.percent_dense}\n'
            f'densification_interval: {self.densification_interval}\n'
            f'opacity_reset_interval: {self.opacity_reset_interval}\n'
            f'densify_from_iter: {self.densify_from_iter}\n'
            f'densify_until_iter: {self.densify_until_iter}\n'
            f'densify_grad_threshold: {self.densify_grad_threshold}\n'
            f'ssim'
        )
        super_extra = super().extra_repr()
        if len(super_extra) > 0:
            _repr = f'{super_extra}\n{_repr}'
        return _repr

if __name__ == '__main__':
    from poketto.datamodule import get_collate_fn
    from poketto.datamodule.datasets import NerfSynthetic
    from poketto.datamodule.data_preprocessors import GSDataPreprocessor
    from torch.utils.data import DataLoader
    dataset = NerfSynthetic(
        data_root='data/nerf_synthetic/lego', pcd_root='data/3dgs/nerf_synthetic'
    )
    collate_fn = get_collate_fn(['img', 'R', 'T', 'FovX', 'FovY'])
    dataloader = DataLoader(dataset, collate_fn=collate_fn)
    preprocessor = GSDataPreprocessor()
    sample = next(iter(dataloader))
    sample = preprocessor(sample)
    print({k: v.shape if k in ['img'] else v for k, v in sample.items()})
    model = GaussianSplatting(sh_degree=3, dataset=dataset)
    model.cuda()
    model.train()
    model(sample)
    print([v.shape for k, v in sample.items() if k in ['pred', 'viewspace_points', 'visibility_filter', 'radii']])
    print(sample['losses'])
