import numpy as np
import torch
import torch.nn.functional as F
from utils import compute_normal_from_depth
import trimesh
from skimage import measure
import logging

trimesh.util.attach_to_log(level=logging.ERROR)


def get_combined_mesh(z, xyz, level=0):
    z = z.astype(np.float32)
    verts, faces, normals, values = measure.marching_cubes(
        volume=z.reshape(xyz[1].shape[0], xyz[0].shape[0], xyz[2].shape[0]).transpose([1, 0, 2]),
        level=level, spacing=(xyz[0][2] - xyz[0][1], xyz[1][2] - xyz[1][1], xyz[2][2] - xyz[2][1]))

    verts = verts + np.array([xyz[0][0], xyz[1][0], xyz[2][0]])

    meshexport = trimesh.Trimesh(verts, faces, normals)

    return meshexport


# 获取推断sdf的ray_o和ray_d，能够设置采样间隔 (sdf不同于nerf，并不要求严格保证每个pixel一根ray)
def get_rays_for_sdf(H, W, intrinsic, c2w, down_scale=1.0):
    device = c2w.device
    h = int(H / down_scale)
    w = int(W / down_scale)
    ys, xs = torch.meshgrid(torch.linspace(0, H - 1, h), torch.linspace(0, W - 1, w))  # pytorch's meshgrid has indexing='ij'
    ys, xs = ys.reshape(-1), xs.reshape(-1)  # [hw]
    ys, xs = ys.to(device), xs.to(device)

    # target camera下，呈像面的3d position
    xs = xs[None, None, :]  # [1,1,hw]
    ys = ys[None, None, :]  # [1,1,hw]
    ones = torch.ones_like(xs)
    xys = torch.cat([xs, ys, ones], dim=1)  # [1,3,hw]
    dirs = torch.inverse(intrinsic) @ xys  # [1,3,hw]
    rays_d = c2w[:, :3, :3] @ dirs  # [1,3,hw]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:, :3, -1].clone()  # [1,3]

    pixel_coordinates = xys[:, 1, :] * W + xys[:, 0, :]  # [B,hw]

    return rays_o, rays_d, pixel_coordinates


def get_ndc_coordinate(w2c_ref, intrinsic_ref, point_samples, inv_scale, near=2., far=6., lindisp=False):
    '''
        point_samples [B,N_rays,N_sample,3]
    '''
    B, N_rays, N_samples, _ = point_samples.shape
    point_samples = point_samples.reshape(B, -1, 3)

    # wrap to ref view
    R = w2c_ref[:, :3, :3]  # [B,3,3]
    T = w2c_ref[:, :3, 3:]  # [B,3,1]
    point_samples = torch.matmul(point_samples, R.transpose(1, 2)) + T.transpose(1, 2)  # [B,N_rays*N_sample,3]

    # using projection [B,N_rays*N_sample,3]x[B,3,3]
    point_samples_pixel = point_samples @ intrinsic_ref.transpose(1, 2)
    # [B,N_rays*N_sample,3]
    point_samples_pixel[:, :, :2] = (point_samples_pixel[:, :, :2] / (point_samples_pixel[:, :, -1:] + 1e-7)) \
                                    / inv_scale.reshape(1, 1, 2)  # normalize to 0~1
    if not lindisp:
        point_samples_pixel[:, :, 2] = (point_samples_pixel[:, :, 2] - near) / (far - near)  # normalize to 0~1
    else:
        point_samples_pixel[:, :, 2] = (1.0 / point_samples_pixel[:, :, 2] - 1.0 / near) / (1.0 / far - 1.0 / near)

    point_samples_pixel = point_samples_pixel.view(B, N_rays, N_samples, 3)
    return point_samples_pixel


# get points position out of the view
def get_invaliable_pos(grid_aligned, H, W, w2cs, intrinsics, near_fars, mask):
    grid_points = grid_aligned['grid_points']
    N_samples = grid_points.shape[0]
    inv_scale = torch.tensor([W - 1, H - 1]).to(w2cs.device)
    points_ndc = get_ndc_coordinate(w2cs, intrinsics, grid_points.reshape(1, N_samples, 1, 3),
                                    inv_scale, near=near_fars[0], far=near_fars[1])
    points_ndc = points_ndc.reshape(N_samples, 3)  # [0~1]
    invalid_xyz1 = (points_ndc <= 0)
    invalid_xyz2 = (points_ndc >= 1)
    invalid_xyz = torch.sum(invalid_xyz1, dim=1) + torch.sum(invalid_xyz2, dim=1)
    invalid_idx = torch.where(invalid_xyz > 0)[0].cpu().numpy()

    # use mask
    if mask is not None:
        R = w2cs[0, :3, :3]  # [B,3,3]
        T = w2cs[0, :3, 3:]  # [B,3,1]
        point_samples = torch.matmul(grid_points, R.transpose(0, 1)) + T.transpose(0, 1)  # [N,3]
        point_samples_pixel = point_samples @ intrinsics[0].transpose(0, 1)
        points_xy = point_samples_pixel[:, :2] / point_samples_pixel[:, 2:3]  # [N,2]
        valid_x = ((W > points_xy[:, 0]).to(torch.long) * (points_xy[:, 0] >= 0).to(torch.long))  # [N]
        valid_y = ((H > points_xy[:, 1]).to(torch.long) * (points_xy[:, 1] >= 0).to(torch.long))
        valid_xy = valid_x + valid_y
        valid_mask_idx = torch.where(valid_xy == 2)[0]
        valid_mask_pos = points_xy[valid_mask_idx].to(torch.long)  # [N,2]
        mask_xy = mask[valid_mask_pos[:, 1], valid_mask_pos[:, 0]]
        invalid_mask_idx = torch.where(mask_xy == 0)[0]
        invalid_mask_idx = valid_mask_idx[invalid_mask_idx].cpu().numpy()
        invalid_idx = np.concatenate([invalid_idx, invalid_mask_idx], axis=0)
        invalid_idx = np.unique(invalid_idx)

    return invalid_idx


def get_grid(points, resolution, device, input_min=None, input_max=None, eps=0.1):
    if input_min is None or input_max is None:
        input_min = torch.min(points, dim=0)[0].squeeze().numpy()
        input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).to(device)
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


def volumetric_fusion(batch, near_fars, input_min, input_max, resolution=100, max_thickness=0.05, N_samples=128,
                      down_scale=1.0, pred_depth=False, normal_weights=False, depth_mask=False):
    imgs = batch['images']
    B, H, W, _ = imgs.shape
    intrinsics = batch['intrinsics']
    w2cs = batch['w2cs']
    c2ws = batch['c2ws']
    h = int(H / down_scale)
    w = int(W / down_scale)
    # infer for each view 取交集mask
    manual_mask = batch['manual_masks'][0]
    device = imgs.device
    if depth_mask:
        mask = batch['gt_depths'][0] > 0
        mask = mask.to(torch.float32)
        mask = mask + manual_mask
        mask[mask < 2] = 0
        mask[mask == 2] = 1
    else:
        mask = manual_mask.to(torch.float32)

    if pred_depth:
        pred_depth = batch['depths_pred'].reshape(1, 1, H, W)
    else:
        pred_depth = batch['gt_depths'].reshape(1, 1, H, W)
    if normal_weights:
        normal = compute_normal_from_depth(pred_depth, torch.inverse(intrinsics[0]), mask=mask.reshape(1, 1, H, W))
        normal = F.interpolate(normal, size=(h, w), mode='nearest')

        rays_o, rays_d, pixel_coordinates = get_rays_for_sdf(H, W, intrinsics, c2ws, down_scale)
        # rays_d world to camera
        rays_d = w2cs[:, :3, :3] @ rays_d
        rays_d = F.normalize(rays_d.reshape(1, 3, h, w), dim=1)
        normal = F.normalize(normal, dim=1)
        weights = torch.sum(rays_d * normal, dim=1) * -1  # [1,h,w]
        weights = torch.clamp(weights, 0, 1).repeat(N_samples, 1, 1)  # [N_samples,h,w]
        weights = weights.permute(1, 2, 0).reshape(h * w, N_samples)

    pred_depth = F.interpolate(pred_depth, size=(h, w), mode='nearest')
    pred_depth = pred_depth.reshape(h * w)

    # 通过depth过滤ray上的sdf
    pred_depth_ = (pred_depth - near_fars[0]) / (near_fars[1] - near_fars[0])  # norm depth to [0~1]
    ray_values = torch.linspace(0., 1., steps=N_samples, device=pred_depth.device).reshape(1, N_samples).repeat(h * w, 1)  # [hw,N_samples]
    sdf_values = (ray_values - pred_depth_.reshape(h * w, 1)) * (-1)
    pred_depth_end = torch.clamp(((pred_depth_ + max_thickness) * N_samples).long(), 0, N_samples - 1).reshape(h * w, 1)  # 转为long
    depth_onehot_end = torch.zeros_like(sdf_values).scatter_(1, pred_depth_end, 1)  # 转onehot [hw,N_samples]
    depth_mask = torch.cumsum(depth_onehot_end, dim=1)
    sdf_values = sdf_values * (1 - depth_mask) + (1e3 * depth_mask)
    sdf_values = sdf_values.reshape(h, w, N_samples)

    grid_aligned = get_grid(None, resolution, device, input_min=input_min, input_max=input_max, eps=0.0)
    grid_points = grid_aligned['grid_points']  # [R*R*R,3]
    invalid_idx = get_invaliable_pos(grid_aligned, H, W, w2cs, intrinsics, near_fars, mask)
    # 为了吧sdf_values塞到sdf_cube里，要获取sdf_cube里每个位置所对应在sdf_values的'位置',
    # 即获取sdf_cube的tgt_ndc_coordinates, sdf_cube的位置为grid_points
    inv_scale = torch.tensor([W - 1, H - 1]).to(imgs.device)
    points_ndc = get_ndc_coordinate(w2cs, intrinsics, grid_points.reshape(1, grid_points.shape[0], 1, 3),
                                    inv_scale, near=near_fars[0], far=near_fars[1])
    points_ndc = points_ndc.reshape(grid_points.shape[0], 3)  # [0~1]
    points_ndc = points_ndc * 2 - 1.0  # [R**3,3] to [-1~1]
    # [1,1,h,w,N_samples]->[1,1,N_samples(z),h(y),w(x)]
    sdf_values = sdf_values.reshape(1, 1, h, w, N_samples).permute(0, 1, 4, 2, 3)
    points_ndc = points_ndc.reshape(1, 1, 1, grid_points.shape[0], 3)
    # sdf_values:[1,1,N_samples(z),h(y),w(x)], points_ndc:[1,1,1,R**3,3] (xyz)
    # sdf_cube:[1,1,1,1,R**3]->[R**3]
    sdf_cube = F.grid_sample(sdf_values, points_ndc, align_corners=True, mode='bilinear', padding_mode='zeros')[0, 0, 0, 0]
    sdf_cube[invalid_idx] = 1e3  # padding outside pts for fusion
    cube_mask = 1 - (sdf_cube == 1e3).to(torch.float32)

    # xyz
    xyz = grid_aligned['xyz']
    sdf_cube = sdf_cube.reshape(xyz[0].shape[0], xyz[1].shape[0], xyz[2].shape[0])

    if normal_weights:
        weight_values = weights.reshape(1, 1, h, w, N_samples).permute(0, 1, 4, 2, 3)
        weight_cube = F.grid_sample(weight_values, points_ndc, align_corners=True, mode='bilinear', padding_mode='zeros')[0, 0, 0, 0]  # [R**3]
        weight_cube = weight_cube.reshape(xyz[0].shape[0], xyz[1].shape[0], xyz[2].shape[0])
    else:
        weight_cube = torch.ones_like(sdf_cube)
    cube_mask = cube_mask.reshape(xyz[0].shape[0], xyz[1].shape[0], xyz[2].shape[0])

    return sdf_cube, xyz, weight_cube, cube_mask
