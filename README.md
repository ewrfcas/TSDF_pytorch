# TSDF pytorch
Simple implementation of TSDF（Truncated Signed Distance Function）in pytorch

### DataSets

MVS images and cameras comes from [here](http://roboimagedata.compute.dtu.dk/?page_id=36). 
Data is preprocessed by [MVSNet](https://github.com/YoYo000/MVSNet). 
Manual masks are from [IDR](https://github.com/lioryariv/idr). 
Depth results used here are from [PatchMatchNet](https://github.com/FangjinhuaWang/PatchmatchNet). 
The normalized parameter `norm_param.pkl` is used to ensure that the target mesh is displayed in a -1 to 1 cube.
Demo data of scan114 can be downloaded in [here](https://drive.google.com/file/d/1nqsIlcPLiAc802rMcQSfYsNTdTcV8uei/view?usp=sharing).

You can also get results of real-world data with cameras and depths. 

### DTU images (not used in fusion)

![img](./assets/img.png)

### DTU depths


| gt depth   | patchmatchnet depth   |
| -------- | -------- |
| <img src="https://github.com/ewrfcas/TSDF_pytorch/blob/main/assets/depth.png" width="400"/> | <img src="https://github.com/ewrfcas/TSDF_pytorch/blob/main/assets/depth2.png" width="400"/> |

### Fusion with pytorch

The volumetric fusion costs about 2s for each view in GPU.

Fusion with predicted depth:
```
CUDA_VISIBLE_DEVICES=0 python run.py --data_dir <your DTU path> --pred_depth_dir <your prediction depth>
```

Fusion with ground truth depth:
```
CUDA_VISIBLE_DEVICES=0 python run.py --use_gt_depth --data_dir <your DTU path>
```

### Fusion results of PatchMatch predicted depth

| 1-view | 2-view | 4-view | 8-view |
| -------- | -------- | -------- | -------- |
| <img src="https://github.com/ewrfcas/TSDF_pytorch/blob/main/assets/snapshot01.png" width="200"/> | <img src="https://github.com/ewrfcas/TSDF_pytorch/blob/main/assets/snapshot02.png" width="200"/> | <img src="https://github.com/ewrfcas/TSDF_pytorch/blob/main/assets/snapshot03.png" width="200"/> | <img src="https://github.com/ewrfcas/TSDF_pytorch/blob/main/assets/snapshot04.png" width="200"/> |




