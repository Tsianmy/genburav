import os
import json
import torch
import itertools
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
from typing import NamedTuple
from torchvision import io as tvio, tv_tensors
from poketto.ops.sh_utils import SH2RGB
from poketto.ops.transforms3d import getWorld2View2, focal2fov, fov2focal
from poketto.utils import glogger
from .base_dataset import BaseDataset

class BasicPointCloud(NamedTuple):
    points : torch.Tensor
    colors : torch.Tensor
    normals : torch.Tensor

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = torch.hstack(cam_centers)
        avg_cam_center = torch.mean(cam_centers, dim=1, keepdim=True)
        center = avg_cam_center
        dist = torch.linalg.vector_norm(cam_centers - center)
        diagonal = torch.amax(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam['R'], cam['T'])
        C2W = torch.inverse(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(
        points=torch.from_numpy(positions),
        colors=torch.from_numpy(colors),
        normals=torch.from_numpy(normals)
    )

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

class NerfSynthetic(BaseDataset):
    def __init__(
        self,
        data_root='',
        pcd_root='',
        transforms=None,
        test_mode=False,
        white_background=False,
        **kwargs
    ):
        self.pcd_root = pcd_root
        self.white_background = white_background
        super().__init__(
            data_root=data_root,
            transforms=transforms,
            test_mode=test_mode,
            **kwargs
        )
    
    def read_cameras(self, ann_file):
        with open(ann_file) as f:
            contents = json.load(f)

        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        extension = '.png'
        data_list = []
        glogger.info(f'read from {ann_file}')
        for idx, frame in enumerate(frames):
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = torch.tensor(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = torch.inverse(c2w)
            # R is stored transposed due to 'glm' in CUDA code
            R = w2c[:3, :3].t()
            T = w2c[:3, 3]

            image_path = os.path.join(self.data_root, frame["file_path"] + extension)
            image_name = Path(image_path).stem
            img = tvio.read_image(image_path)
            
            alpha = img[3] / 255.0

            bg = 255 if self.white_background else 0

            img = img[:3] * alpha + bg  * (1 - alpha)

            fovy = focal2fov(fov2focal(fovx, img.shape[2]), img.shape[1])

            data_list.append(dict(
                sample_idx=idx, R=R, T=T, FovY=fovy, FovX=fovx, img=tv_tensors.Image(img),
                image_path=image_path, image_name=image_name,
                width=img.shape[2], height=img.shape[1]
            ))
        return data_list
    
    def get_point_cloud(self):
        os.makedirs(self.pcd_root, exist_ok=True)
        ply_path = os.path.join(self.pcd_root, "points3d.ply")
        if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
            num_pts = 100_000
            glogger.info(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = torch.random((num_pts, 3)) * 2.6 - 1.3
            shs = torch.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=torch.zeros((num_pts, 3))
            )

            storePly(ply_path, xyz.numpy(), SH2RGB(shs.numpy()) * 255)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
        return pcd

    def load_data_list(self):
        if self.test_mode:
            ann_files = [os.path.join(self.data_root, 'transforms_test.json')]
        else:
            ann_files = [
                os.path.join(self.data_root, 'transforms_train.json'),
                os.path.join(self.data_root, 'transforms_test.json')
            ]
        data_list = list(itertools.chain.from_iterable(
            [self.read_cameras(file) for file in ann_files]
        ))

        glogger.info('get camera norm')
        self.nerf_normalization = getNerfppNorm(data_list)
        glogger.info(self.nerf_normalization)

        if self.test_mode:
            self.pcd = None
        else:
            glogger.info('get point cloud')
            self.pcd = self.get_point_cloud()
            glogger.info(f'point num: {len(self.pcd.points)}')

        return data_list

    def raw_data(self, idx):
        return self.data_list[idx]

if __name__ == '__main__':
    dataset = NerfSynthetic(
        data_root='data/nerf_synthetic/lego', pcd_root='data/3dgs/nerf_synthetic'
    )
    print(len(dataset))
    print({k: v.shape if k in ['R', 'T', 'img'] else v for k, v in dataset[0].items()})
    print({k: v.shape if k in ['R', 'T', 'img'] else v for k, v in dataset[-1].items()})
    print(dataset.nerf_normalization)
    print(dataset.pcd.points.shape)