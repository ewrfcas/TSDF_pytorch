import argparse
from dataset import NormalizedMVSDTU
from torch.utils.data import DataLoader
import numpy as np
from utils import to_cuda
from fusion import volumetric_fusion, get_combined_mesh
from tqdm import tqdm
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='../DTU_MVSNet', help='input data directory')
parser.add_argument("--output_dir", type=str, default='results', help='output directory')
parser.add_argument("--pred_depth_dir", type=str, default=None, help='predicted depth map dir')
parser.add_argument("--near", type=float, default=1.1066, help='nearest plane distance')
parser.add_argument("--far", type=float, default=2.3565, help='farest plane distance')
parser.add_argument("--max_view_num", type=int, default=10, help='view number used in fusion')
parser.add_argument("--resolution", type=int, default=300, help='resolution for the matching cubes')
parser.add_argument("--down_scale", type=float, default=1.0, help='downsample for rays of each image')
parser.add_argument("--max_thickness", type=float, default=0.15, help='truncated distance')
parser.add_argument("--N_samples", type=int, default=256, help='point number of each rays')
parser.add_argument("--scan", type=str, default='scan114', help='scan no of DTU')
parser.add_argument("--with_cpu", default=False, action='store_true')
parser.add_argument("--use_gt_depth", default=False, action='store_true')
parser.add_argument("--no_normal_weight", default=False, action='store_true')

args = parser.parse_args()

if __name__ == '__main__':
    val_dataset = NormalizedMVSDTU(root_dir=args.data_dir, pred_depth_dir=args.pred_depth_dir,
                                   scan=args.scan, max_len_per=args.max_view_num)
    data_loader = DataLoader(val_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=False)

    total_sdfs = torch.zeros([args.resolution, args.resolution, args.resolution], dtype=torch.float32, device='cpu' if args.with_cpu else 'cuda')
    count_masks = torch.zeros([args.resolution, args.resolution, args.resolution], dtype=torch.float32, device='cpu' if args.with_cpu else 'cuda')
    xyz = None
    for sample in tqdm(data_loader):
        if not args.with_cpu:
            sample = to_cuda(sample)
        if args.use_gt_depth:
            pred_depth = False
            depth_mask = True
        else:
            pred_depth = True
            depth_mask = False
        sdf_values, xyz, weights, valid_mask = volumetric_fusion(sample, near_fars=[args.near, args.far],
                                                                 input_min=np.array([-1., -1., -1.]), input_max=np.array([1., 1., 1.]),
                                                                 resolution=args.resolution, max_thickness=args.max_thickness,
                                                                 N_samples=args.N_samples, down_scale=args.down_scale, pred_depth=pred_depth,
                                                                 normal_weights=False if args.no_normal_weight else True, depth_mask=depth_mask)
        count_masks += valid_mask
        valid_sdf = sdf_values * valid_mask * weights
        total_sdfs += valid_sdf

    count_masks[count_masks > 0] = 1
    total_sdfs = total_sdfs * count_masks + 1e3 * (1 - count_masks)
    total_sdfs = total_sdfs.cpu().numpy()
    mesh = get_combined_mesh(total_sdfs, xyz, level=0)
    components = mesh.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=np.float32)
    mesh_clean = components[areas.argmax()]
    os.makedirs(f'{args.output_dir}', exist_ok=True)
    mesh_clean.export(f'{args.output_dir}/{args.scan}.ply')
