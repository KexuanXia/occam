import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp
from torch import nn
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch.hash import HashTable
import matplotlib.pyplot as plt

# read point cloud and drop the 4th column since intensity is not necessary for voxelization
source_file_path = 'demo_data/000001.bin'
if source_file_path.split('.')[-1] == 'bin':
    points = np.fromfile(source_file_path, dtype=np.float32)
    points = points.reshape(-1, 4)[:, :3]
elif source_file_path.split('.')[-1] == 'npy':
    points = np.load(source_file_path)[:, :3]
else:
    raise NotImplementedError

# number of total points
print("number of points: ", points.shape[0])

# transfer np into torch
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
points = torch.from_numpy(points).float().to(device)

# initialize PointToVoxel
point_to_voxel = PointToVoxel(
    vsize_xyz=[0.2, 0.2, 0.2],  # voxel size
    coors_range_xyz=[-80, -80, -10, 80, 80, 50],  # coordinate ranges
    num_point_features=3,  # number of point features
    max_num_voxels=20000,  # maximum voxels
    max_num_points_per_voxel=200,  # maximum points in each voxel
    device=torch.device("cuda:0")  # GPU
)

voxels, indices, num_points_per_voxel = point_to_voxel(points)

# print("Voxels Shape:", voxels.shape)
# print("Indices Shape:", indices.shape)
# print("Number of Points per Voxel:", num_points_per_voxel)
# print("Number of Points per Voxel:", num_points_per_voxel.tolist())
print("Number of Points after Voxelization:", sum(num_points_per_voxel.tolist()))
zero_voxels_count = (num_points_per_voxel == 0).sum().item()
print("Number of zero voxels:", zero_voxels_count)
voxel_stats = pd.DataFrame(num_points_per_voxel.tolist(), columns=['points_per_voxel'])
print(voxel_stats.describe())


# print("Example of Voxel Data:\n", voxels[1])  # 显示第一个体素的数据
# print("Example of Indices Data:\n", indices[1])  # 显示第一个体素的位置索引
# print("Example of Num Points per Voxel:\n", num_points_per_voxel[1])  # 显示第一个体素的点数

