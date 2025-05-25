import math
import torch

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def getWorld2View2(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    ndim = R.ndim
    if ndim == 2:
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
    Rt = torch.zeros((R.shape[0], 4, 4))
    Rt[:, :3, :3] = R.transpose(1, 2)
    Rt[:, :3, 3] = t
    Rt[:, 3, 3] = 1.0

    C2W = torch.inverse(Rt)
    cam_center = C2W[:, :3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:, :3, 3] = cam_center
    Rt = torch.inverse(C2W)
    if ndim == 2:
        Rt = Rt.squeeze()
    return Rt.float()

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = torch.tan((fovY / 2))
    tanHalfFovX = torch.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(fovX.shape[0], 4, 4)

    z_sign = 1.0

    P[:, 0, 0] = 2.0 * znear / (right - left)
    P[:, 1, 1] = 2.0 * znear / (top - bottom)
    P[:, 0, 2] = (right + left) / (right - left)
    P[:, 1, 2] = (top + bottom) / (top - bottom)
    P[:, 3, 2] = z_sign
    P[:, 2, 2] = z_sign * zfar / (zfar - znear)
    P[:, 2, 3] = -(zfar * znear) / (zfar - znear)
    return P