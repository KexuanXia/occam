"""
    For evaluating 3D heat maps, we need to remove some points and inference again to see how the
    model performance drops.
    This file include the point removal, re-inference of the evaluation.
"""


import pickle
import numpy as np
import torch
from google.protobuf import text_format

import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.pytorch.models import fusion
import open3d as o3d
from scipy.spatial.transform import Rotation

from train import (build_inference_net,
                   example_convert_to_torch,
                   get_inference_input_dict,
                   predict_kitti_to_anno)


# inference after point removal
def occam_evaluation_inference(start_idx, end_idx, it_nr=3000, save_result=False,
                               config_path='/home/xkx/CLOCs/second/configs/car.fhd.config',
                               second_model_dir='../model_dir/second_model',
                               fusion_model_dir='../CLOCs_SecCas_pretrained'):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    # model configuration
    model_cfg = config.model.second
    detection_2d_path = config.train_config.detection_2d_path
    center_limit_range = model_cfg.post_center_limit_range
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)
    class_names = target_assigner.classes
    net = build_inference_net(config_path, second_model_dir)
    fusion_layer = fusion.fusion()
    fusion_layer.cuda()
    net.cuda()
    torchplus.train.try_restore_latest_checkpoints(fusion_model_dir, [fusion_layer])
    net.eval()
    fusion_layer.eval()

    # inference
    for idx in range(start_idx, end_idx):
        idx_str = str(idx).zfill(6)
        input_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_croped_by_occam/{idx_str}.bin'
        i_path = f'/home/xkx/kitti/training/image_2/{idx_str}.png'
        attr_map_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/heat_map/{idx_str}_{it_nr}.pkl'

        info = read_kitti_info_val(idx=idx)
        input_pc = np.fromfile(input_path, dtype=np.float32)
        input_pc = input_pc.reshape(-1, 4)

        base_det_boxes, base_det_labels, base_det_scores = read_original_dt_results(idx_str)
        with open(attr_map_path, 'rb') as file:
            attr_map = pickle.load(file)
        print(f"attr_map.shape: {attr_map.shape}")
        print(f"base_det_boxes: {base_det_boxes}")

        results = []
        if attr_map.shape[0] == 0:
            print("This scene doesn't contain any target.")
        else:
            # remove points in descending, random, ascending order
            sorted_points_desc, sorted_points_random, sorted_points_asc = (
                filter_and_sort_points_by_importance(input_pc, base_det_boxes, attr_map))
            removal_results_desc = progressively_remove_points(input_pc, sorted_points_desc)
            removal_results_random = progressively_remove_points(input_pc, sorted_points_random)
            removal_results_asc = progressively_remove_points(input_pc, sorted_points_asc)
            removal_results = [removal_results_desc, removal_results_random, removal_results_asc]

            # visualize_point_cloud_and_bboxes(removal_results[0][50], base_det_boxes)
            for removal_result in removal_results:
                result = []
                for percentage, remaining_points in removal_result.items():
                    example = get_inference_input_dict(config=config,
                                                       voxel_generator=voxel_generator,
                                                       target_assigner=target_assigner,
                                                       info=info,
                                                       points=remaining_points,
                                                       i_path=i_path)
                    example = example_convert_to_torch(example, torch.float32)

                    with torch.no_grad():
                        dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                            net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                            model_cfg.lidar_input)
                        prediction_dicts = prediction_dicts[0]
                        result.append(prediction_dicts)
                results.append(result)
        print(f"results: {results}")

        # save detection results
        if save_result:
            save_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_evaluation/{idx_str}.pkl'
            with open(save_path, 'wb') as output_file:
                pickle.dump(results, output_file)
            print(f"detection result of dropped {idx_str}.bin have been saved")


# visualization code, just for debugging and understanding
def visualize_point_cloud_and_bboxes(input_pc, base_det_boxes):
    # creat Open3D objects
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(input_pc[:, :3])

    # Defines a list for storing geometries
    geometries = [point_cloud]

    # Traverse the detection box and create a cube
    for box in base_det_boxes:
        # Creating an Open3D OrientedBoundingBox
        rot_mat = Rotation.from_rotvec([0, 0, box[6]]).as_matrix()
        bb = o3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
        bb.color = (1.0, 0.0, 1.0)

        # Add the cube to the geometry list
        geometries.append(bb)

    # Create a visualizer and set the background color
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)

    # Set the background color
    vis.get_render_option().background_color = np.ones(3) * 0.25
    vis.get_render_option().point_size = 4.0

    # Run the visualizer
    vis.run()
    vis.destroy_window()


# filter which points are within bounding boxes
# sort the filtered points according to the heat map
def filter_and_sort_points_by_importance(input_pc, base_det_boxes, attr_map):
    """
    Filter out the points in the rotated detection box and sort them by importance. Return three sets of results at the same time:
    1. Points and their indices sorted in descending order of importance.
    2. Points and their indices sorted randomly.
    3. Points and their indices sorted in ascending order of importance.

    :param input_pc: numpy array, shape (M, 3), representing M 3D points (x, y, z).
    :param base_det_boxes: numpy array, shape (N, 7), representing N detection boxes,
    Each box is defined by the center point (cx, cy, cz), size (length, width, height) and rotation angle (yaw).
    :param attr_map: numpy array, shape (N, M), representing the importance of each point in each detection box.
    :return: tuple of three lists, each list contains three different sorting results, sorted in descending, random and ascending order.
    """
    sorted_points_desc = []
    sorted_points_random = []
    sorted_points_asc = []

    for i, box in enumerate(base_det_boxes):
        cx, cy, cz, length, width, height, yaw = box

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, sin_yaw, 0],
            [-sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        # Move the point cloud to a coordinate system with the center of the detection box as the origin
        translated_points = input_pc[:, :3] - np.array([cx, cy, cz])

        # Apply a rotation matrix to transform the point cloud to the detection box coordinate system
        rotated_points = np.dot(translated_points, rotation_matrix.T)

        # Calculate the boundaries of the detection box
        x_min, x_max = -length / 2, length / 2
        y_min, y_max = -width / 2, width / 2
        z_min, z_max = -height / 2, height / 2

        # Find the points within the detection box
        inside_mask = (
                (rotated_points[:, 0] >= x_min) & (rotated_points[:, 0] <= x_max) &
                (rotated_points[:, 1] >= y_min) & (rotated_points[:, 1] <= y_max) &
                (rotated_points[:, 2] >= z_min) & (rotated_points[:, 2] <= z_max)
        )

        print(f"number of points in boxes: {sum(inside_mask)}")

        # point_indices = np.where(inside_mask)[0]

        # Extract the points in the detection box and their corresponding importance
        points_in_box = input_pc[inside_mask]
        importance_in_box = attr_map[i, inside_mask]

        # Sort by importance in descending order
        sorted_indices_desc = np.argsort(importance_in_box)[::-1]
        sorted_points_desc.append(
            list(zip(points_in_box[sorted_indices_desc], importance_in_box[sorted_indices_desc]))
        )

        # random
        random_indices = np.random.permutation(len(importance_in_box))
        sorted_points_random.append(
            list(zip(points_in_box[random_indices], importance_in_box[random_indices]))
        )

        # ascending
        sorted_indices_asc = np.argsort(importance_in_box)
        sorted_points_asc.append(
            list(zip(points_in_box[sorted_indices_asc], importance_in_box[sorted_indices_asc]))
        )

    return sorted_points_desc, sorted_points_random, sorted_points_asc


# remove points with the top 0%, 10%... to 100% importance gradually
def progressively_remove_points(input_pc, sorted_points_with_importance):
    """
    Gradually remove the points with the top 0%, 10%... to 100% importance.

    :param input_pc: numpy array, shape (M, 3), representing M three-dimensional points (x, y, z).
    :param sorted_points_with_importance: list of lists, each list contains the points and their importance in the corresponding detection box, and is sorted in descending order of importance.
    :return: dict, key is the percentage of removal, value is the array of remaining points.
    """
    removal_results = {}

    # Calculate the amount that needs to be removed
    for percent in range(0, 110, 10):
        remaining_points = input_pc.copy()

        # Remove from each detection box
        for box_index, points_with_importance in enumerate(sorted_points_with_importance):
            points_in_box_count = len(points_with_importance)
            # if there is no point in the bbox, there is no need to remove any points
            if points_in_box_count != 0:
                points_to_remove = int(points_in_box_count * (percent / 100))

                sorted_points, _ = zip(*points_with_importance)
                sorted_points = np.array(sorted_points)
                mask = np.ones(len(remaining_points), dtype=bool)

                for point in sorted_points[:points_to_remove]:
                    point_indices = np.where((remaining_points == point).all(axis=1))[0]
                    if point_indices.size > 0:
                        mask[point_indices[0]] = False

                remaining_points = remaining_points[mask]

        removal_results[percent] = remaining_points

    return removal_results


def read_kitti_info_val(idx):
    file_path = "/home/xkx/kitti/kitti_infos_trainval.pkl"
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    for item in data:
        if item.get('image_idx') == idx:
            return item
    return IndexError


def read_original_dt_results(idx_str):
    #read_path = f'/home/xkx/kitti/training/velodyne_masked_dt_results/{source_file_path[-10: -4]}_original.pkl'
    read_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_original_dt_results/{idx_str}_original.pkl'
    with open(read_path, 'rb') as file:
        data = pickle.load(file)
    pred_boxes = data["box3d_lidar"]
    pred_boxes[:, [3, 4]] = pred_boxes[:, [4, 3]]
    pred_scores = data["scores"]
    pred_labels = data["label_preds"] + 1

    pred_boxes = pred_boxes.cpu().numpy()
    pred_scores = pred_scores.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()

    for i in range(pred_boxes.shape[0]):
        pred_boxes[i, 6] = -pred_boxes[i, 6] - np.pi / 2
        pred_boxes[i, 2] = pred_boxes[i, 2] + pred_boxes[i, 5] / 2

    print(f"Successfully read original detection results from {read_path}")

    return pred_boxes, pred_labels, pred_scores



"""
    -------------------------------------------------------------------------------
    I tried to do the global evaluation for 3D case, but it didn't work.
    The following codes are for global evaluation but it was deprecated.
"""


def sort_voxels_by_importance_for_each_object(input_pc, attr_map, pt_vx_id):
    """
    对每个检测对象，根据体素的重要度对点云进行排序，提供三种排序方式：
    1. 按体素重要度降序排序。
    2. 随机排序。
    3. 按体素重要度升序排序。

    :param input_pc: numpy array，形状为 (N, 3)，表示 N 个三维点 (x, y, z)。
    :param attr_map: numpy array，形状为 (M, N)，表示 M 个检测对象，每个对象对应 N 个点的重要度。
    :param pt_vx_id: numpy array，形状为 (N,)，表示 N 个点所属的体素 ID。
    :return: tuple of three lists，每个列表包含 M 个子列表，每个子列表对应一个检测对象，
             包含按不同排序方式排序的 (voxel_id, voxel_importance) 元组。
    """
    M, N = attr_map.shape
    # 确保 attr_map 的列数与点云一致
    assert N == len(input_pc), "attr_map 的列数必须与 input_pc 的长度一致"

    # 初始化结果列表
    sorted_voxels_desc_list = []
    sorted_voxels_random_list = []
    sorted_voxels_asc_list = []

    # 获取所有唯一体素的 ID
    unique_voxels = np.unique(pt_vx_id)

    for i in range(M):
        # 获取第 i 个检测对象的所有点的重要度
        importance_scores = attr_map[i, :]

        # 计算每个体素的平均重要度
        voxel_importance = {}
        for vx in unique_voxels:
            voxel_points_indices = np.where(pt_vx_id == vx)[0]
            voxel_importance[vx] = np.mean(importance_scores[voxel_points_indices])

        # 将体素和它们的平均重要度转换为列表
        voxel_importance_list = [(vx, voxel_importance[vx]) for vx in voxel_importance]

        # 按重要度降序排序
        sorted_voxels_desc = sorted(voxel_importance_list, key=lambda x: x[1], reverse=True)
        sorted_voxels_desc_list.append(sorted_voxels_desc)

        # 随机排序
        np.random.shuffle(voxel_importance_list)
        sorted_voxels_random_list.append(voxel_importance_list)

        # 按重要度升序排序
        sorted_voxels_asc = sorted(voxel_importance_list, key=lambda x: x[1])
        sorted_voxels_asc_list.append(sorted_voxels_asc)

    return sorted_voxels_desc_list, sorted_voxels_random_list, sorted_voxels_asc_list


def remove_voxels_by_percentage(input_pc, attr_map, pt_vx_id):
    """
    对每个检测对象，按照三种排序方式，按百分比移除体素及其对应的点，并返回移除后的点云。

    :param input_pc: numpy array，形状为 (N, 3)，表示 N 个三维点 (x, y, z)。
    :param attr_map: numpy array，形状为 (M, N)，表示 M 个检测对象，每个对象对应 N 个点的重要度。
    :param pt_vx_id: numpy array，形状为 (N,)，表示 N 个点所属的体素 ID。
    :param percentages: list，指定移除的百分比，比如 [0, 10, 20, ...]。
    :return: list of list，每个检测对象对应一个子列表，
             每个子列表包含3个子列表，分别对应3种排序方式。
             每种排序方式又包含多个子列表，表示按不同百分比移除体素后的剩余点云。
    """
    sorted_voxels_desc_list, sorted_voxels_random_list, sorted_voxels_asc_list = (
        sort_voxels_by_importance_for_each_object(input_pc, attr_map, pt_vx_id))

    M, N = attr_map.shape
    percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    result = []

    # 遍历每个检测对象
    for i in range(M):
        remaining_points_by_desc = []
        remaining_points_by_random = []
        remaining_points_by_asc = []

        # 获取每种排序方式下的体素
        sorted_voxels_desc = sorted_voxels_desc_list[i]
        sorted_voxels_random = sorted_voxels_random_list[i]
        sorted_voxels_asc = sorted_voxels_asc_list[i]
        print(len(sorted_voxels_desc), len(sorted_voxels_random), len(sorted_voxels_asc))

        # 遍历每个百分比，计算移除后的点
        for percentage in percentages:
            # 计算需要移除的体素数量
            num_voxels_to_remove = int(len(sorted_voxels_desc) * percentage / 100)

            # 按降序移除体素
            removed_voxel_ids_desc = [vx_id for vx_id, _ in sorted_voxels_desc[:num_voxels_to_remove]]
            remaining_points_desc = input_pc[~np.isin(pt_vx_id, removed_voxel_ids_desc)]
            remaining_points_by_desc.append(remaining_points_desc)

            # 按随机顺序移除体素
            removed_voxel_ids_random = [vx_id for vx_id, _ in sorted_voxels_random[:num_voxels_to_remove]]
            remaining_points_random = input_pc[~np.isin(pt_vx_id, removed_voxel_ids_random)]
            remaining_points_by_random.append(remaining_points_random)

            # 按升序移除体素
            removed_voxel_ids_asc = [vx_id for vx_id, _ in sorted_voxels_asc[:num_voxels_to_remove]]
            remaining_points_asc = input_pc[~np.isin(pt_vx_id, removed_voxel_ids_asc)]
            remaining_points_by_asc.append(remaining_points_asc)

        # 保存每个检测对象下的3种排序方式的结果
        result.append([
            remaining_points_by_desc,   # 重要度降序
            remaining_points_by_random, # 随机排序
            remaining_points_by_asc     # 重要度升序
        ])

    return result


def progressively_remove_points_globally(input_pc, sorted_points_with_importance):
    """
    逐步移除重要度排在前0%、10%...到100%的点。

    :param input_pc: numpy array, 形状为 (M, 3)，表示 M 个三维点 (x, y, z)。
    :param sorted_points_with_importance: list of lists，每个列表包含位于对应检测框内的点及其重要度，并按重要度降序排序。
    :return: dict, key为移除的百分比，value为剩余点的数组。
    """
    removal_results = []

    num_objects = len(sorted_points_with_importance)
    num_points = input_pc.shape[0]
    percentages = np.arange(0, 1.1, 0.1)
    num_points_to_remove = (num_points * percentages).astype(int)

    for i in range(num_objects):
        removal_results_single_object = []
        for num_points_to_remove_ in num_points_to_remove:
            remaining_points = input_pc.copy()
            remaining_points = remaining_points[num_points_to_remove_:]
            removal_results_single_object.append(remaining_points)
        removal_results.append(removal_results_single_object)

    return removal_results


def occam_evaluation_inference_globally(start_idx, end_idx, it_nr=3000, save_result=False,
                                        config_path='/home/xkx/CLOCs/second/configs/car.fhd.config',
                                        second_model_dir='../model_dir/second_model',
                                        fusion_model_dir='../CLOCs_SecCas_pretrained'):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    model_cfg = config.model.second
    detection_2d_path = config.train_config.detection_2d_path
    center_limit_range = model_cfg.post_center_limit_range
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg, bv_range, box_coder)
    class_names = target_assigner.classes
    net = build_inference_net(config_path, second_model_dir)
    fusion_layer = fusion.fusion()
    fusion_layer.cuda()
    net.cuda()
    torchplus.train.try_restore_latest_checkpoints(fusion_model_dir, [fusion_layer])
    net.eval()
    fusion_layer.eval()

    sort_types = ['descend', 'random', 'ascend']
    for idx in range(start_idx, end_idx):
        idx_str = str(idx).zfill(6)
        print(f"Evaluating {idx_str}")
        input_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_croped_by_occam/{idx_str}.bin'
        i_path = f'/home/xkx/kitti/training/image_2/{idx_str}.png'
        attr_map_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/heat_map/{idx_str}_{it_nr}.pkl'
        pt_vx_id_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/pt_vx_id/{idx_str}.npy'
        pt_vx_id = np.load(pt_vx_id_path)
        print(len(pt_vx_id))
        print(f"pt_vx_id: {pt_vx_id}")

        info = read_kitti_info_val(idx=idx)
        input_pc = np.fromfile(input_path, dtype=np.float32)
        input_pc = input_pc.reshape(-1, 4)

        base_det_boxes, base_det_labels, base_det_scores = read_original_dt_results(idx_str)
        with open(attr_map_path, 'rb') as file:
            attr_map = pickle.load(file)
        num_objects = attr_map.shape[0]

        eval_dt_results = {
            'descend': [],
            'random': [],
            'ascend': []
        }
        if num_objects == 0:
            print("This scene doesn't contain any target.")
        else:
            removal_results_list = remove_voxels_by_percentage(input_pc, attr_map, pt_vx_id)

            for object_index in range(num_objects):
                print(f"Evaluating the {object_index}/{num_objects} object")
                removal_results = {
                    'descend': removal_results_list[object_index][0],
                    'random': removal_results_list[object_index][1],
                    'ascend': removal_results_list[object_index][2]
                }
                for sort_type in sort_types:
                    print(sort_type)
                    list_of_removal_results = removal_results[sort_type]
                    dt_results = []
                    for percentage, remaining_points in enumerate(list_of_removal_results):
                        print(percentage)
                        # visualize_point_cloud_and_bboxes(list_of_removal_results[percentage], base_det_boxes)
                        example = get_inference_input_dict(config=config,
                                                           voxel_generator=voxel_generator,
                                                           target_assigner=target_assigner,
                                                           info=info,
                                                           points=remaining_points,
                                                           i_path=i_path)
                        example = example_convert_to_torch(example, torch.float32)

                        with torch.no_grad():
                            dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                                net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                                model_cfg.lidar_input)
                            prediction_dicts = prediction_dicts[0]
                            print(f"bbox: {prediction_dicts['box3d_lidar'].cpu().numpy()}")
                            print(f"scores: {prediction_dicts['scores'].cpu().numpy()}")
                            dt_results.append(prediction_dicts)
                    eval_dt_results[sort_type].append(dt_results)
        # print(eval_dt_results)
        if save_result:
            save_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_evaluation_dt_results_globally'
                         f'/{idx_str}.pkl')
            with open(save_path, 'wb') as output_file:
                pickle.dump(eval_dt_results, output_file)
            print(f"detection result of {idx_str} for global evaluation have been saved")





if __name__ == '__main__':
    occam_evaluation_inference(1370, 1371, save_result=False)
    # occam_evaluation_inference_globally(0, 100, save_result=True)