"""
    This file is just for visualization and debugging.
"""


import numpy as np
import open3d as o3d

source_file_path = '../../kitti/training/velodyne/000006.bin'
# source_file_path = '/home/xkx/kitti/training/velodyne_masked/000002/000002.bin'
# source_file_path = '/home/xkx/Stereo-3D-Detection/results/generated_lidar/velodyne/000000.bin'
# source_file_path = '../Pseudo_Lidar_V2/results/sdn_kitti_train_set/pseudo_lidar_trainval/000007.bin'
# source_file_path = '../Pseudo_Lidar_V2/results/sdn_kitti_train_set/pseudo_lidar_trainval_sparse/000007.bin'
# source_file_path = '../Pseudo_Lidar_V2/results/sdn_kitti_train_set/ptc_from_corrected_depthmap/000007.bin'
# source_file_path = '../Pseudo_Lidar_V2/results/sdn_kitti_train_set_download/pseudo_lidar_trainval_sparse/000007.bin'
if source_file_path.split('.')[-1] == 'bin':
    points = np.fromfile(source_file_path, dtype=np.float32)
    points = points.reshape(-1, 4)[:, :3]
elif source_file_path.split('.')[-1] == 'npy':
    points = np.load(source_file_path)[:, :3]
else:
    raise NotImplementedError

def visualize_points(points, z_plane):
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Convert NumPy array to Open3D PointCloud
    pcd.points = o3d.utility.Vector3dVector(points)  # Assuming points have at least 3 coordinates

    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Set the background color and point size
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.array([0.25, 0.25, 0.25])

    # Optionally draw the origin axis
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    if z_plane is not None:
        plane_points = np.array([
            [30, -10, z_plane],
            [30, 10, z_plane],
            [50, -10, z_plane],
            [50, 10, z_plane]
        ])
        plane_lines = [[0, 1], [1, 3], [3, 2], [2, 0]]
        plane_colors = [[0, 1, 0] for _ in range(len(plane_lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(plane_points)
        line_set.lines = o3d.utility.Vector2iVector(plane_lines)
        line_set.colors = o3d.utility.Vector3dVector(plane_colors)
        vis.add_geometry(line_set)

    # Run the visualizer
    vis.run()
    vis.destroy_window()

# The function call is commented out to follow the guidelines of not executing in PCI
visualize_points(points, -1.83)


