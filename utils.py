import os, torch, cv2, re
import numpy as np
import torch.nn.functional as F


def pixel_points(batch, height, width, device):
    '''
    Given width and height, creates a mesh grid, and returns homogeneous
    coordinates
    of image in a 3 x W*H Tensor

    Arguments:
        width {Number} -- Number representing width of pixel grid image
        height {Number} -- Number representing height of pixel grid image

    Returns:
        torch.Tensor -- 1x2xHxW, oriented in x, y order
    '''
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                           torch.arange(0, width, dtype=torch.float32, device=device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]

    return xyz


def compute_normal_from_depth(depth, K_inv, xyz=None, mask=None, apply_mask=True):
    # depth,mask:[B,1,H,W]
    if mask is None:
        mask = torch.tensor((depth > 0), device=depth.device)
    else:
        mask = mask.to(torch.bool)

    B, _, H, W = depth.shape

    if xyz is None:
        xyz = pixel_points(B, H, W, depth.device)

    batch, _, height, width = depth.shape
    # [B,3,3]x[B,3,HW]->[B,3,HW]->[B,3,H,W]
    cam_r = (K_inv @ xyz).reshape(batch, 3, height, width)
    cam_p = cam_r * depth  # [B,3,H,W]

    pxd = F.pad(depth, [1, 1, 0, 0])
    pyd = F.pad(depth, [0, 0, 1, 1])
    px = F.pad(cam_p, [1, 1, 0, 0])
    py = F.pad(cam_p, [0, 0, 1, 1])
    pxm = F.pad(mask, [1, 1, 0, 0])
    pym = F.pad(mask, [0, 0, 1, 1])

    # compute valid mask for normal
    mx1 = (pxm[:, :, :, 2:] & pxm[:, :, :, 1:-1])
    mx2 = (pxm[:, :, :, :-2] & pxm[:, :, :, 1:-1])
    my1 = (pym[:, :, 2:] & pym[:, :, 1:-1])
    my2 = (pym[:, :, :-2] & pym[:, :, 1:-1])

    # filter mask by limited depth diff
    ddx1 = (pxd[:, :, :, 2:] - pxd[:, :, :, 1:-1])
    ddx2 = (pxd[:, :, :, 1:-1] - pxd[:, :, :, :-2])
    ddy1 = (pyd[:, :, 2:] - pyd[:, :, 1:-1])
    ddy2 = (pyd[:, :, 1:-1] - pyd[:, :, :-2])
    if (apply_mask):
        mx1 = mx1 & (ddx1.abs() < pxd[:, :, :, 1:-1] * 0.05)
        mx2 = mx2 & (ddx2.abs() < pxd[:, :, :, 1:-1] * 0.05)
        my1 = my1 & (ddy1.abs() < pyd[:, :, 1:-1] * 0.05)
        my2 = my2 & (ddy2.abs() < pyd[:, :, 1:-1] * 0.05)
    mx = mx1 | mx2
    my = my1 | my2
    m = mx & my

    # compute finite diff gradients
    dx1 = (px[:, :, :, 2:] - px[:, :, :, 1:-1])
    dx2 = (px[:, :, :, 1:-1] - px[:, :, :, :-2])
    dy1 = (py[:, :, 2:] - py[:, :, 1:-1])
    dy2 = (py[:, :, 1:-1] - py[:, :, :-2])
    # dxs = dx1 * mx1 + dx2 * (mx2 & ~mx1)
    # dys = dy1 * my1 + dy2 * (my2 & ~my1)
    dxs = (dx1 * mx1 + dx2 * mx2) / (mx1.float() + mx2.float() + (~mx).float())
    dys = (dy1 * my1 + dy2 * my2) / (my1.float() + my2.float() + (~my).float())
    dx = F.normalize(dxs, dim=1, p=2)
    dy = F.normalize(dys, dim=1, p=2)

    # compute normal direction from cross products
    normal = torch.cross(dy, dx, dim=1)

    # flip normal based on camera view
    normal = F.normalize(normal, p=2, dim=1)
    cam_dir = F.normalize(cam_r, p=1, dim=1)

    dot = (cam_dir * normal).sum(1, keepdim=True)
    normal *= -dot.sign()

    if (apply_mask):
        nm = m.repeat(1, 3, 1, 1)
        normal[~nm] = 0

    return normal


def to_cuda(batch):
    for k in batch:
        if type(batch[k]) == torch.Tensor:
            batch[k] = batch[k].cuda()
    return batch


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale
