"""
    This file realizes the systematic combination strategy.
"""


import numpy as np
import pickle
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from google.protobuf import text_format
import torch

import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.pytorch.models import fusion
import matplotlib.pyplot as plt
import open3d as o3d

from train import (build_inference_net,
                   example_convert_to_torch,
                   get_inference_input_dict,
                   predict_kitti_to_anno,
                   read_kitti_info_val)

from d_rise import iou, gen_cam, mask_image


# These masks were originally stored in groups of 8, and now they are separated
def expand_mask(masked_input_path):
    with open(masked_input_path, 'rb') as file:
        pc_masks = pickle.load(file)
    expand_pc_masks = []
    for item in pc_masks:
        points = item['points']
        for batch in range(8):
            filtered_points = points[points[:, 0] == batch, 1:]
            expand_pc_masks.append(filtered_points)
    expand_pc_masks = np.array(expand_pc_masks)
    return expand_pc_masks


# Run the inference to get the detection results
def systematically_mask_combination_inference(start_idx, end_idx, save_result=False,
                                              config_path='/home/xkx/CLOCs/second/configs/car.fhd.config',
                                              second_model_dir='../model_dir/second_model',
                                              fusion_model_dir='../CLOCs_SecCas_pretrained'):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    # model configuration
    model_cfg = config.model.second
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

    for idx in range(start_idx, end_idx):
        idx_str = str(idx).zfill(6)
        pc_mask_indexes_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_heat_map_mask_indexes_2/'
                                f'{idx_str}_30.pkl')
        with open(pc_mask_indexes_path, 'rb') as file:
            pc_mask_indexes = pickle.load(file)

        num_objects = pc_mask_indexes.shape[0]
        if num_objects == 0:
            continue

        image_mask_indexes_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/'
                                   f'D_RISE_heat_map_important_mask_indexes_2/'
                                   f'{idx_str}_30.pkl')
        with open(image_mask_indexes_path, 'rb') as file:
            image_mask_indexes = pickle.load(file)
        if num_objects != image_mask_indexes.shape[0]:
            print("number of objects in heat maps are different! Check!")

        it_nr = 3000
        masked_input_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_masked_pointcloud_2/{idx_str}_{it_nr}.pkl'
        pc_masks = expand_mask(masked_input_path)

        # how many masks are you going to use
        number_of_filter = 30

        important_pc_masks_index = pc_mask_indexes[:, :number_of_filter]
        unimportant_pc_masks_index = pc_mask_indexes[:, -number_of_filter:][:, ::-1]

        important_pc_masks = pc_masks[important_pc_masks_index]
        unimportant_pc_masks = pc_masks[unimportant_pc_masks_index]

        important_image_masks_index = image_mask_indexes[:, :number_of_filter]
        unimportant_image_masks_index = image_mask_indexes[:, -number_of_filter:][:, ::-1]

        info = read_kitti_info_val(idx=idx)
        i_path = f'/home/xkx/kitti/training/image_2/{idx_str}.png'

        detection_results = {
            'il_ii': [],  # important lidar masks and important image masks # 30*30=900
            'il_ui': [],  # important lidar masks and unimportant image masks
            'ul_ii': [],  # unimportant lidar masks and important image masks
            'ul_ui': []  # unimportant lidar masks and unimportant image masks
        }

        # Iterate through all targets
        for object_index in range(num_objects):
            il_ii = []
            il_ui = []
            # Traversing 30 important masks
            for masked_pc in important_pc_masks[object_index]:
                example = get_inference_input_dict(config=config,
                                                   voxel_generator=voxel_generator,
                                                   target_assigner=target_assigner,
                                                   info=info,
                                                   points=masked_pc,
                                                   i_path=i_path)
                example = example_convert_to_torch(example, torch.float32)

                # Traverse 30 important and 30 unimportant masks
                for important_index in important_image_masks_index[object_index]:
                    detection_2d_path = (f"/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/"
                                         f"CasCade_2d_detection_results_3000/{idx_str}/results/{important_index}.txt")
                    with torch.no_grad():
                        dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                            net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                            model_cfg.lidar_input, flag_2d=True)
                        prediction_dicts = prediction_dicts[0]
                    il_ii.append(prediction_dicts)

                for unimportant_index in unimportant_image_masks_index[object_index]:
                    detection_2d_path = (f"/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/"
                                         f"CasCade_2d_detection_results_3000/{idx_str}/results/{unimportant_index}.txt")
                    with torch.no_grad():
                        dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                            net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                            model_cfg.lidar_input, flag_2d=True)
                        prediction_dicts = prediction_dicts[0]
                    il_ui.append(prediction_dicts)

            detection_results['il_ii'].append(il_ii)
            detection_results['il_ui'].append(il_ui)

            ul_ii = []
            ul_ui = []
            # Traverse 30 unimportant masks
            for masked_pc in unimportant_pc_masks[object_index]:
                example = get_inference_input_dict(config=config,
                                                   voxel_generator=voxel_generator,
                                                   target_assigner=target_assigner,
                                                   info=info,
                                                   points=masked_pc,
                                                   i_path=i_path)
                example = example_convert_to_torch(example, torch.float32)

                # Traverse 30 important and 30 unimportant masks
                for important_index in important_image_masks_index[object_index]:
                    detection_2d_path = (f"/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/"
                                         f"CasCade_2d_detection_results_3000/{idx_str}/results/{important_index}.txt")
                    with torch.no_grad():
                        dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                            net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                            model_cfg.lidar_input, flag_2d=True)
                        prediction_dicts = prediction_dicts[0]
                    ul_ii.append(prediction_dicts)

                for unimportant_index in unimportant_image_masks_index[object_index]:
                    detection_2d_path = (f"/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/"
                                         f"CasCade_2d_detection_results_3000/{idx_str}/results/{unimportant_index}.txt")
                    with torch.no_grad():
                        dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                            net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                            model_cfg.lidar_input, flag_2d=True)
                        prediction_dicts = prediction_dicts[0]
                    ul_ui.append(prediction_dicts)

            detection_results['ul_ii'].append(ul_ii)
            detection_results['ul_ui'].append(ul_ui)

        if save_result:
            save_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/systematically_mask_combination_2/'
                         f'{idx_str}.pkl')
            with open(save_path, 'wb') as output_file:
                pickle.dump(detection_results, output_file)
            print(f"systematically_mask_combination {idx_str} have been saved")


def read_original_dt_results_2d(idx_str):
    read_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_original_dt_results/'
                 f'{idx_str}_original.pkl')
    with open(read_path, 'rb') as file:
        data = pickle.load(file)
    pred_boxes = data["bbox"]
    pred_scores = data["scores"]
    pred_labels = data["label_preds"] + 1

    pred_boxes = pred_boxes.cpu().numpy()
    pred_scores = pred_scores.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()

    print(f"Successfully read 2d original detection results from {read_path}")

    return pred_boxes, pred_labels, pred_scores


def get_image(idx):
    str_idx = str(idx).zfill(6)
    image_path = f'/home/xkx/kitti/training/image_2/{str_idx}.png'
    image = cv2.imread(image_path)
    return image


def get_original_pointcloud(idx):
    str_idx = str(idx).zfill(6)
    source_file_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/'
                        f'velodyne_croped_by_occam/{str_idx}.bin')
    if source_file_path.split('.')[-1] == 'bin':
        points = np.fromfile(source_file_path, dtype=np.float32)
        points = points.reshape(-1, 4)
    elif source_file_path.split('.')[-1] == 'npy':
        points = np.load(source_file_path)
    else:
        raise NotImplementedError

    return points


def get_masked_pointcloud(idx):
    str_idx = str(idx).zfill(6)
    source_file_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/'
                        f'velodyne_masked_pointcloud_2/{str_idx}_3000.pkl')
    with open(source_file_path, 'rb') as file:
        masked_pt_list = pickle.load(file)

    masked_pt = []
    mask_itself = []

    # Traverse the masked_pt_list and extract the 'points' and 'mask' fields respectively
    for item in masked_pt_list:
        masked_pt.append(item['points'])
        mask_itself.append(item['mask'])

    pt_keep = []
    remove_batch_mask_itself = []

    # Traverse each two-dimensional array
    for batch_pt, batch_mask in zip(masked_pt, mask_itself):
        # Iterate through the group numbers, from 0 to 7
        for group in range(8):
            # Filter out the data rows belonging to the current group
            group_pt = batch_pt[batch_pt[:, 0] == group]
            group_mask = batch_mask[group]

            # Delete the first column, which is the information about the group, and keep the remaining 4 columns
            group_pt_without_group = group_pt[:, 1:]

            # Add the processed data to a new list
            pt_keep.append(group_pt_without_group)
            remove_batch_mask_itself.append(group_mask)

    original_pt = get_original_pointcloud(idx)
    pt_removed = []
    for mask in remove_batch_mask_itself:
        points = original_pt[np.logical_not(mask)]
        pt_removed.append(points)

    return pt_removed, pt_keep


def get_masked_voxel(idx):
    str_idx = str(idx).zfill(6)
    source_file_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/'
                        f'velodyne_masked_pointcloud_2/{str_idx}_3000.pkl')
    with open(source_file_path, 'rb') as file:
        masked_pt_list = pickle.load(file)
    vx_coord = []
    vx_keep_ids = []

    for dict in masked_pt_list:
        for key, val in dict.items():
            if key == 'vx_orig_coord':
                for i in range(8):
                    batch_mask = [val[:, 0] == i]
                    vx_coord.append(val[batch_mask][:, 1:])
            elif key == 'vx_keep_ids':
                for i in range(8):
                    batch_mask = [val[:, 0] == i]
                    vx_keep_ids.append(val[batch_mask][:, 1])

    vx_keep = []
    for vx, keep_id in zip(vx_coord, vx_keep_ids):
        vx_keep.append(vx[keep_id, :])

    return vx_keep


def get_masked_image(idx):
    mask_path = (f'//media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CasCade_2d_detection_results_3000'
                 f'/{str(idx).zfill(6)}/masks/masks_3000.pkl')
    with open(mask_path, 'rb') as file:
        masks = pickle.load(file)
    return masks


# read calibration matrix from KITTI dataset
def read_calib_info(idx):
    path = f'/home/xkx/kitti/training/calib/{str(idx).zfill(6)}.txt'

    data_dict = {}

    with open(path, 'r') as file:
        for line in file:
            # Remove newlines and split on colons
            line = line.strip()
            if ':' in line:
                key, values = line.split(':', 1)
                # Remove the spaces before and after the key and parse the corresponding value
                key = key.strip()
                values = np.fromstring(values, sep=' ')
                data_dict[key] = values
    P2 = data_dict['P2'].reshape(3, 4)
    R0 = data_dict['R0_rect'].reshape(3, 3)
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = R0
    Tvelo2cam = data_dict['Tr_velo_to_cam'].reshape(3, 4)
    Tvelo2cam = np.vstack((Tvelo2cam, np.array([0, 0, 0, 1])))
    return P2, R0_rect, Tvelo2cam


# Treat the 3D voxel as a 3D detection box and project it onto the image plane
def voxel_to_box_2d(box_3d, projection_matrix):
    # voxel size
    l, w, h = 0.2, 0.2, 0.2
    yaw = 0

    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    boxes_2d = []

    # The offset of the eight vertices of the 3D detection box relative to the center point
    offsets = np.array([
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1]
    ]) / 2

    for i in range(box_3d.shape[0]):
        # Extract the center point of the box and its dimensions
        center = box_3d[i, :3]

        # Calculate vertex coordinates
        corners = np.dot(offsets * [l, w, h], R.T) + center

        # Transform to camera coordinate system and project to image plane
        corners_hom = np.hstack((corners, np.ones((8, 1))))
        projected = (projection_matrix @ corners_hom.T).T

        # Normalization
        projected[:, :2] /= projected[:, 2:3]

        # Get the minimum and maximum coordinates
        x_min, y_min = np.min(projected[:, :2], axis=0)
        x_max, y_max = np.max(projected[:, :2], axis=0)

        boxes_2d.append([x_min, y_min, x_max, y_max])

    return np.array(boxes_2d)


def get_important_masks_index(idx):
    lidar_mask_index_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_heat_map_mask_indexes_2'
                             f'/{str(idx).zfill(6)}_30.pkl')
    with open(lidar_mask_index_path, 'rb') as file:
        lidar_mask_indexes = pickle.load(file)

    image_mask_index_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/D_RISE_heat_map_important_mask_indexes_2'
                             f'/{str(idx).zfill(6)}_30.pkl')
    with open(image_mask_index_path, 'rb') as file:
        image_mask_indexes = pickle.load(file)

    return lidar_mask_indexes, image_mask_indexes


# visualization, only for debugging and understanding when developing
def visualization_lidar(idx):
    str_idx = str(idx).zfill(6)
    pt_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_croped_by_occam/'
               f'{str_idx}.bin')
    image_path = f'/home/xkx/kitti/training/image_2/{str_idx}.png'
    input_pc = np.fromfile(pt_path, dtype=np.float32).reshape(-1, 4)
    pt_removed, pt_keep = get_masked_pointcloud(idx)
    pt_to_show = pt_keep[0]

    # 创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(input_pc[:, :3])

    geometries = [point_cloud]

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

    input_pc = pt_to_show

    # 创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(input_pc[:, :3])

    geometries = [point_cloud]

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


# visualization, only for debugging and understanding when developing
def visualization(image, box_2d, image_mask, common_mask):
    # 创建三个图像的副本
    image_box_blackout = image.copy()
    image_mask_blackout = image.copy()

    # 第一张图：将 box_2d 内的像素设为黑色
    for box in box_2d:
        x1, y1, x2, y2 = map(int, box)
        image_box_blackout[y1:y2, x1:x2] = 0

    # 第二张图：将 image_mask 中为 False 的位置设为黑色
    image_mask = image_mask >= 0.5
    image_mask_blackout[~image_mask] = 0

    common_mask_blackout = mask_image(image, common_mask)

    # 使用 OpenCV 显示图像
    cv2.imshow("Box Blackout", image_box_blackout)
    cv2.imshow("Image Mask Blackout", image_mask_blackout)
    cv2.imshow("Common Mask Blackout", common_mask_blackout)

    # 等待按键事件并关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# compute the common area between 2D mask and projected 3D mask
def compute_common_area(image_h, image_w, projection_matrix, lidar_mask, image_mask):
    # list of 3000, each has voxels which are kept
    box_2d = voxel_to_box_2d(lidar_mask, projection_matrix)
    lidar_mask = np.zeros((image_h, image_w), dtype=bool)

    for box in box_2d:
        x1, y1, x2, y2 = map(int, box)
        lidar_mask[y1:y2, x1:x2] = True

    image_mask = image_mask >= 0.5
    common_mask = np.logical_or(lidar_mask, image_mask)
    # common_mask = image_mask * lidar_mask
    return common_mask


# generate the unified heat map by systematic combination
def generate_unified_heatmap(idx,
                          save_figures=False,
                          save_heatmap=False,
                          show=False):
    base_det_boxes, base_det_labels, base_det_scores = read_original_dt_results_2d(str(idx).zfill(6))
    num_of_objects = base_det_boxes.shape[0]
    if num_of_objects == 0:
        heat_map = np.zeros((0, 0, 0))
    else:
        i_path = f'/home/xkx/kitti/training/image_2/{str(idx).zfill(6)}.png'
        image = cv2.imread(i_path)
        image_h, image_w = image.shape[:2]
        detection_results_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/systematically_mask_combination_2'
                                  f'/{str(idx).zfill(6)}.pkl')
        with open(detection_results_path, 'rb') as file:
            # detection_results = {
            #     'il_ii': [],     # important lidar masks and important image masks # 30*30=900
            #     'il_ui': [],     # important lidar masks and unimportant image masks
            #     'ul_ii': [],     # unimportant lidar masks and important image masks
            #     'ul_ui': []      # unimportant lidar masks and unimportant image masks
            # }
            detection_results = pickle.load(file)

        # Initialize heat map
        heat_map = np.zeros((num_of_objects, image_h, image_w), dtype=np.float32)

        # read lidar mask
        lidar_masks = get_masked_voxel(idx)

        # read image mask
        image_masks = get_masked_image(idx)

        # read lidar and image mask sorted indexes
        number_of_filter = 30
        lidar_mask_indexes, image_mask_indexes = get_important_masks_index(idx)

        important_pc_masks_index = lidar_mask_indexes[:, :number_of_filter]
        unimportant_pc_masks_index = lidar_mask_indexes[:, -number_of_filter:][:, ::-1]

        important_image_masks_index = image_mask_indexes[:, :number_of_filter]
        unimportant_image_masks_index = image_mask_indexes[:, -number_of_filter:][:, ::-1]

        P2, R0_rect, Tvelo2cam = read_calib_info(idx)
        projection_matrix = P2 @ R0_rect @ Tvelo2cam

        # Generate heat map for each detected object in the image
        for target_index in range(num_of_objects):
            target_box = base_det_boxes[target_index]
            print(f'target: {target_box}')
            single_detection = []
            single_detection.extend(detection_results['il_ii'][target_index])
            single_detection.extend(detection_results['il_ui'][target_index])
            single_detection.extend(detection_results['ul_ii'][target_index])
            single_detection.extend(detection_results['ul_ui'][target_index])
            sampling_map = np.zeros((image_h, image_w))

            index_to_show = [0, 100, 200, 900, 1000, 1100, 1900, 2000, 2100, 2800, 2900, 3000]
            scores = np.zeros(3600)
            for index, detection in enumerate(single_detection):
                print(f'index: {index}/3600')
                type_index, lidar_mask_index, image_mask_index = (
                    index // 900, (index % 900) // 30, (index % 900) % 30)
                if type_index == 0:
                    lidar_mask = lidar_masks[important_pc_masks_index[target_index, lidar_mask_index]]
                    image_mask = image_masks[important_image_masks_index[target_index, image_mask_index]]
                elif type_index == 1:
                    lidar_mask = lidar_masks[important_pc_masks_index[target_index, lidar_mask_index]]
                    image_mask = image_masks[unimportant_image_masks_index[target_index, image_mask_index]]
                elif type_index == 2:
                    lidar_mask = lidar_masks[unimportant_pc_masks_index[target_index, lidar_mask_index]]
                    image_mask = image_masks[unimportant_image_masks_index[target_index, image_mask_index]]
                elif type_index == 3:
                    lidar_mask = lidar_masks[unimportant_pc_masks_index[target_index, lidar_mask_index]]
                    image_mask = image_masks[important_image_masks_index[target_index, image_mask_index]]
                else:
                    print('type index error')

                common_mask = compute_common_area(image_h, image_w, projection_matrix, lidar_mask, image_mask)

                pred_boxes = detection["bbox"]
                pred_scores = detection["scores"]
                pred_boxes = pred_boxes.cpu().numpy()
                pred_scores = pred_scores.cpu().numpy()
                print(pred_boxes)
                print(pred_scores)

                # if index in index_to_show:
                #     visualization(image, [], image_mask, common_mask)

                score = 0
                for box, score_value in zip(pred_boxes, pred_scores):
                    current_score = iou(target_box, box) * score_value
                    if current_score > score:
                        score = current_score
                scores[index] = score
                sampling_map += common_mask
                heat_map[target_index] += common_mask * score
                # heat_map[target_index] += (1-common_mask) * score

            heat_map[target_index] /= sampling_map
            print(heat_map[target_index])
            image_with_bbox, _ = gen_cam(image, heat_map[target_index])
            left_corner = target_box[:2].astype(np.int32)
            right_corner = target_box[2:].astype(np.int32)
            cv2.rectangle(image_with_bbox, left_corner, right_corner,
                          (0, 0, 255), 5)
            if save_figures:
                save_path = (f"/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/systematically_mask_combination_heat_map/"
                             f"{str(idx).zfill(6)}_{target_index}.png")
                cv2.imwrite(save_path, image_with_bbox)
                print(f"{str(idx).zfill(6)}_{target_index}.png has been saved")

            if show:
                cv2.imshow('image', image_with_bbox)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    if save_heatmap:
        save_path = (f"/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/systematically_mask_combination_heat_map/"
                     f"{str(idx).zfill(6)}.pkl")
        with open(save_path, 'wb') as output_file:
            pickle.dump(heat_map, output_file)
        print(f"heat map has been saved: {heat_map}")
    return heat_map


if __name__ == '__main__':
    generate_unified_heatmap(9, save_figures=True, save_heatmap=False, show=True)

    # idx = 6
    # i_path = f'/home/xkx/kitti/training/image_2/{str(idx).zfill(6)}.png'
    # image = cv2.imread(i_path)
    # image_h, image_w = image.shape[:2]
    #
    # # read lidar mask
    # lidar_masks = get_masked_voxel(idx)
    #
    # # read image mask
    # image_masks = get_masked_image(idx)
    #
    # # read lidar and image mask sorted indexes
    # number_of_filter = 30
    # lidar_mask_indexes, image_mask_indexes = get_important_masks_index(idx)
    #
    # important_pc_masks_index = lidar_mask_indexes[:, :number_of_filter]
    # unimportant_pc_masks_index = lidar_mask_indexes[:, -number_of_filter:][:, ::-1]
    #
    # important_image_masks_index = image_mask_indexes[:, :number_of_filter]
    # unimportant_image_masks_index = image_mask_indexes[:, -number_of_filter:][:, ::-1]
    #
    # P2, R0_rect, Tvelo2cam = read_calib_info(idx)
    # projection_matrix = P2 @ R0_rect @ Tvelo2cam
    #
    # for i in range(3600):
    #     lidar_mask = lidar_masks[i]
    #     image_mask = image_masks[i]
    #
    #     box_2d = voxel_to_box_2d(lidar_mask, projection_matrix)
    #     common_mask = compute_common_area(image_h, image_w, projection_matrix, lidar_mask, image_mask)
    #     visualization(image, box_2d, image_mask, common_mask)