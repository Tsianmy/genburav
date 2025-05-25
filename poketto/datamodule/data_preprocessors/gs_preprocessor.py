from poketto.ops.transforms3d import getWorld2View2, getProjectionMatrix
from .base import BaseDataPreprocessor

class GSDataPreprocessor(BaseDataPreprocessor):
    def __init__(
        self,
        *args,
        znear=0.1,
        zfar=100.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.minmax = True
        self.znear = znear
        self.zfar = zfar
    
    def __call__(self, data: dict, training=False):
        img = self.to_cuda(data['img'])

        img = img.float()
        if self.minmax:
            img.div_(255.)
            data['minmax'] = True
        data['img'] = img

        data['FovX'] = data['FovX'].float()
        data['FovY'] = data['FovY'].float()
        data['world_view_transform'] = self.to_cuda(
            getWorld2View2(data['R'], data['T']).transpose(1, 2)
        )
        data['projection_matrix'] = self.to_cuda(
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=data['FovX'], fovY=data['FovY']
            ).transpose(1, 2)
        )
        data['full_proj_transform'] = self.to_cuda(
            data['world_view_transform'].bmm(data['projection_matrix'])
        )
        data['camera_center'] = self.to_cuda(
            data['world_view_transform'].inverse()[:, 3, :3]
        )

        return data
    