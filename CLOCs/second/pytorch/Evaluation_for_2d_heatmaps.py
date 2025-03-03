"""
    For evaluating 2D heat maps, we need to remove some pixels and inference again to see how the
    model performance drops.
    This file include the pixel removal, re-inference of the evaluation.
    Note that local evaluation failed for 2D case, so we did global evaluation as well.
"""


import os.path
import pickle

import cv2
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
import matplotlib.pyplot as plt
import open3d as o3d

from train import (build_inference_net,
                   example_convert_to_torch,
                   get_inference_input_dict,
                   predict_kitti_to_anno,
                   read_kitti_info_val)
from pathlib import Path


# inference after removing pixels, locally
def d_rise_inference_for_evaluation(start_idx, end_idx, it_nr=3000, save_result=False,
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

    # inference
    for idx in range(start_idx, end_idx):
        idx_str = str(idx).zfill(6)
        folder_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/D_RISE_Evaluation/'
                       f'CasCade_2d_detection_results/{idx_str}')
        # If the corresponding 2D detection result does not exist, it means there is no target in the image, and skip this loop directly
        if not os.path.exists(folder_path):
            continue

        input_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_croped_by_occam/'
                      f'{idx_str}.bin')
        i_path = f'/home/xkx/kitti/training/image_2/{idx_str}.png'

        info = read_kitti_info_val(idx=idx)
        input_pc = np.fromfile(input_path, dtype=np.float32)
        input_pc = input_pc.reshape(-1, 4)

        example = get_inference_input_dict(config=config,
                                           voxel_generator=voxel_generator,
                                           target_assigner=target_assigner,
                                           info=info,
                                           points=input_pc,
                                           i_path=i_path)
        example = example_convert_to_torch(example, torch.float32)

        detection_results = {
            'descend': [],
            'random': [],
            'ascend': []
        }
        sort_types = ['descend', 'random', 'ascend']

        # Traverse 3 orders
        for sort_type in sort_types:
            for percentage in range(0, 101, 10):
                detection_2d_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/'
                                     f'D_RISE_Evaluation/CasCade_2d_detection_results/'
                                     f'{idx_str}/{sort_type}/{percentage}.txt')
                with torch.no_grad():
                    dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                        net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                        model_cfg.lidar_input, flag_2d=True)
                    prediction_dicts = prediction_dicts[0]
                    detection_results[sort_type].append(prediction_dicts)

        if save_result:
            save_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/D_RISE_Evaluation/CLOC_detection_results/'
                         f'{idx_str}_{it_nr}.pkl')
            with open(save_path, 'wb') as output_file:
                pickle.dump(detection_results, output_file)
            print(f"Dict for D-RISE evaluation for {idx_str}.png have been saved")


# inference after removing pixels, globally
def d_rise_inference_for_evaluation_globally(start_idx, end_idx, it_nr=3000, save_result=False,
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
        folder_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/D_RISE_Evaluation_globally/'
                       f'CasCade_2d_detection_results/{idx_str}')
        if not os.path.exists(folder_path):
            continue

        # need to know how many objects in one scene
        items = os.listdir(folder_path)
        subfolders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
        num_objects = len(subfolders)

        input_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_croped_by_occam/'
                      f'{idx_str}.bin')
        i_path = f'/home/xkx/kitti/training/image_2/{idx_str}.png'

        info = read_kitti_info_val(idx=idx)
        input_pc = np.fromfile(input_path, dtype=np.float32)
        input_pc = input_pc.reshape(-1, 4)

        example = get_inference_input_dict(config=config,
                                           voxel_generator=voxel_generator,
                                           target_assigner=target_assigner,
                                           info=info,
                                           points=input_pc,
                                           i_path=i_path)
        example = example_convert_to_torch(example, torch.float32)

        detection_results_list = []
        sort_types = ['descend', 'random', 'ascend']

        # traverse objects because global evaluation cannot be run parallely in one scene
        for object_index in range(num_objects):
            detection_results_dict = {
                'descend': [],
                'random': [],
                'ascend': []
            }
            for sort_type in sort_types:
                for percentage in range(0, 101, 10):
                    detection_2d_path = f'{folder_path}/object_{object_index}/{sort_type}/{percentage}.txt'
                    with torch.no_grad():
                        dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                            net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                            model_cfg.lidar_input, flag_2d=True)
                        prediction_dicts = prediction_dicts[0]
                        detection_results_dict[sort_type].append(prediction_dicts)
            detection_results_list.append(detection_results_dict)

        print(detection_results_list)
        if save_result:
            save_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/D_RISE_Evaluation_globally/CLOC_detection_results/'
                         f'{idx_str}_{it_nr}.pkl')
            with open(save_path, 'wb') as output_file:
                pickle.dump(detection_results_list, output_file)
            print(f"Dict for D-RISE evaluation for {idx_str}.png have been saved")


def adjust_bounding_box(boxes, image_width, image_height):
    boxes = np.array(boxes)

    boxes[:, 0] = np.clip(boxes[:, 0], 0, image_width)  # 调整 x_min
    boxes[:, 1] = np.clip(boxes[:, 1], 0, image_height)  # 调整 y_min
    boxes[:, 2] = np.clip(boxes[:, 2], 0, image_width)  # 调整 x_max
    boxes[:, 3] = np.clip(boxes[:, 3], 0, image_height)  # 调整 y_max

    return boxes


def iou(box1, box2):
    if len(box1) == 0 or len(box2) == 0:
        return 0
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
    br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
    intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
    area1 = np.prod(box1[2:] - box1[:2])
    area2 = np.prod(box2[2:] - box2[:2])
    return intersection / (area1 + area2 - intersection)


# post-processing inference results for local evaluation
def d_rise_evaluation_result(start_idx, end_idx, it_nr=3000):
    sort_types = ['descend', 'random', 'ascend']
    root_path = '/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/D_RISE_Evaluation/CLOC_detection_results/'
    multiple_score_dict = {
        'descend': [],
        'random': [],
        'ascend': []
    }
    multiple_iou_dict = {
        'descend': [],
        'random': [],
        'ascend': []
    }
    # Calculate the average score and iou of multiple detection boxes in a single scene
    for idx in range(start_idx, end_idx):
        score_dict = {
            'descend': [1],
            'random': [1],
            'ascend': [1]
        }
        iou_dict = {
            'descend': [1],
            'random': [1],
            'ascend': [1]
        }
        path = root_path + str(idx).zfill(6) + '_' + str(it_nr) + '.pkl'
        if not os.path.exists(path):
            print('no pkl')
            continue
        with open(path, 'rb') as file:
            eval_dt_dict = pickle.load(file)
        original_dt_boxes = eval_dt_dict['descend'][0]['bbox'].cpu().numpy()
        original_dt_scores = eval_dt_dict['descend'][0]['scores'].cpu().numpy()
        image_path = f'/home/xkx/kitti/training/image_2/{str(idx).zfill(6)}.png'
        image = cv2.imread(image_path)
        image_h, image_w = image.shape[:2]
        original_dt_boxes = adjust_bounding_box(original_dt_boxes, image_w, image_h)

        num_objects = original_dt_boxes.shape[0]
        if num_objects == 0:
            print('no objects')
            continue

        for sort_type in sort_types:
            dt_res_list = eval_dt_dict[sort_type]
            # 10% to 100%
            for i in range(1, 11):
                bboxes = dt_res_list[i]['bbox'].cpu().numpy()
                bboxes = adjust_bounding_box(bboxes, image_w, image_h)
                scores = dt_res_list[i]['scores'].cpu().numpy()
                temp_IoU = np.zeros(num_objects)
                temp_scores = np.zeros(num_objects)

                for object_index, original_dt_box in enumerate(original_dt_boxes):
                    for eval_object_index, eval_dt_box in enumerate(bboxes):
                        temppp_IoU = iou(original_dt_box, eval_dt_box)
                        if temppp_IoU > temp_IoU[object_index]:
                            temp_IoU[object_index] = temppp_IoU
                            index_max_IoU = eval_object_index
                    temp_scores[object_index] = scores[index_max_IoU] / original_dt_scores[object_index] if \
                    temp_IoU[object_index] > 0.5 else 0
                    temp_scores[object_index] = 1 if temp_scores[object_index] > 1 else temp_scores[object_index]
                    # if temp_IoU[object_index] == 0:
                    #     temp_scores[object_index] = 0
                    # else:
                    #     temp_scores[object_index] = eval_dt_scores[index_max_IoU] / original_dt_scores[object_index]
                score_dict[sort_type].append(np.mean(temp_scores))
                iou_dict[sort_type].append(np.mean(temp_IoU))
            multiple_score_dict[sort_type].append(score_dict[sort_type])
            multiple_iou_dict[sort_type].append(iou_dict[sort_type])

    # Process multiple scenes and take the average
    for (sort_type, scores), (sort_type, ious) in (
            zip(multiple_score_dict.items(), multiple_iou_dict.items())):
        scores, ious = np.array(scores), np.array(ious)
        scores = np.mean(scores, axis=0)
        ious = np.mean(ious, axis=0)
        multiple_score_dict[sort_type] = scores
        multiple_iou_dict[sort_type] = ious
    print(multiple_score_dict)
    print(multiple_iou_dict)
    return multiple_score_dict, multiple_iou_dict


# post-processing inference results for global evaluation
def d_rise_evaluation_result_globally(start_idx, end_idx,
                                      it_nr=3000, iou_threshold=0.5):
    sort_types = ['descend', 'random', 'ascend']
    root_path = '/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/D_RISE_Evaluation_globally/CLOC_detection_results/'
    score_dict_scenes = {
        'descend': [],
        'random': [],
        'ascend': []
    }
    iou_dict_scenes = {
        'descend': [],
        'random': [],
        'ascend': []
    }
    # Calculate the average score and iou of multiple detection boxes in a single scene
    for idx in range(start_idx, end_idx):
        path = root_path + str(idx).zfill(6) + '_' + str(it_nr) + '.pkl'
        if not os.path.exists(path):
            print('no pkl')
            continue
        with open(path, 'rb') as file:
            detection_results_list = pickle.load(file)

        num_objects_1 = len(detection_results_list)
        num_objects_2 = detection_results_list[0]['descend'][0]['bbox'].cpu().numpy().shape[0]
        num_objects = min(num_objects_1, num_objects_2)
        print(num_objects_1, num_objects_2)
        if num_objects == 0:
            print('no objects')
            continue
        score_dict_objects = {
            'descend': [],
            'random': [],
            'ascend': []
        }
        iou_dict_objects = {
            'descend': [],
            'random': [],
            'ascend': []
        }
        for object_index in range(num_objects):
            detection_results_dict = detection_results_list[object_index]
            original_dt_boxes = detection_results_dict['descend'][0]['bbox'].cpu().numpy()
            original_dt_scores = detection_results_dict['descend'][0]['scores'].cpu().numpy()
            image_path = f'/home/xkx/kitti/training/image_2/{str(idx).zfill(6)}.png'
            image = cv2.imread(image_path)
            image_h, image_w = image.shape[:2]
            original_dt_boxes = adjust_bounding_box(original_dt_boxes, image_w, image_h)
            # Take out an object to be calculated from the detection results
            original_dt_score = original_dt_scores[object_index]
            original_dt_box = original_dt_boxes[object_index]

            for sort_type in sort_types:
                temp_scores, temp_IoU = np.zeros(11), np.zeros(11)
                temp_scores[0], temp_IoU[0] = 1, 1
                dt_res_list = detection_results_dict[sort_type]
                for i in range(1, 11):
                    bboxes = dt_res_list[i]['bbox'].cpu().numpy()
                    bboxes = adjust_bounding_box(bboxes, image_w, image_h)
                    scores = dt_res_list[i]['scores'].cpu().numpy()

                    for eval_object_index, eval_dt_box in enumerate(bboxes):
                        temppp_IoU = iou(original_dt_box, eval_dt_box)
                        if temppp_IoU > temp_IoU[i]:
                            temp_IoU[i] = temppp_IoU
                            index_max_IoU = eval_object_index

                    if temp_IoU[i] > iou_threshold:
                        temp_scores[i] = scores[index_max_IoU] / original_dt_score
                    else:
                        temp_scores[i] = 0

                    temp_scores[i] = 1 if temp_scores[i] > 1 else temp_scores[i]

                score_dict_objects[sort_type].append(temp_scores)
                iou_dict_objects[sort_type].append(temp_IoU)

        # Handling multiple objects in the same scene
        for (sort_type, scores), (sort_type, ious) in (
                zip(score_dict_objects.items(), iou_dict_objects.items())):
            scores, ious = np.array(scores), np.array(ious)
            scores = np.mean(scores, axis=0)
            ious = np.mean(ious, axis=0)
            score_dict_objects[sort_type] = scores
            iou_dict_objects[sort_type] = ious

        for sort_type in sort_types:
            score_dict_scenes[sort_type].append(score_dict_objects[sort_type])
            iou_dict_scenes[sort_type].append(iou_dict_objects[sort_type])

    # Process multiple scenes and take the average
    for (sort_type, scores), (sort_type, ious) in (
            zip(score_dict_scenes.items(), iou_dict_scenes.items())):
        scores, ious = np.array(scores), np.array(ious)
        scores = np.mean(scores, axis=0)
        ious = np.mean(ious, axis=0)
        score_dict_scenes[sort_type] = scores
        iou_dict_scenes[sort_type] = ious
    print(score_dict_scenes)
    print(iou_dict_scenes)
    return score_dict_scenes, iou_dict_scenes


# plot evaluation results
def plot_d_rise_evaluation(score_dict, iou_dict):
    x = np.arange(0.0, 1.1, 0.1)
    colors = ['green', 'red', 'blue']
    labels = ['Descend', 'Random', 'Ascend']
    sort_types = ['descend', 'random', 'ascend']

    plt.figure(figsize=(10, 8))
    plt.suptitle('2D Heat Map Evaluation')
    plt.subplot(1, 2, 1)
    for i in range(3):
        plt.plot(x, score_dict[sort_types[i]], color=colors[i], label=labels[i])
        plt.scatter(x, score_dict[sort_types[i]], color=colors[i])
    plt.xlabel('Removed Pixels')
    plt.ylabel('Mean confidence score')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(3):
        plt.plot(x, iou_dict[sort_types[i]], color=colors[i], label=labels[i])
        plt.scatter(x, iou_dict[sort_types[i]], color=colors[i])

    plt.xlabel('Removed Pixels')
    plt.ylabel('Mean IoU')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 显示图形
    plt.show()


def plot_threshold_comparison(score_dict_05, score_dict_07):
    x = np.arange(0.0, 1.1, 0.1)
    colors = ['green', 'red', 'blue']
    labels = ['Descend', 'Random', 'Ascend']
    sort_types = ['descend', 'random', 'ascend']

    plt.figure(figsize=(10, 8))
    plt.title('Mean Score Comparison between Different IoU Threshold')
    for i in range(3):
        plt.plot(x, score_dict_05[sort_types[i]], color=colors[i], label=f'threshold=0.5,{labels[i]}', linestyle='-')
        plt.scatter(x, score_dict_05[sort_types[i]], color=colors[i])

    for i in range(3):
        plt.plot(x, score_dict_07[sort_types[i]], color=colors[i], label=f'threshold=0.7,{labels[i]}', linestyle='--')
        plt.scatter(x, score_dict_07[sort_types[i]], color=colors[i])
    plt.xlabel('Removed Pixels')
    plt.ylabel('Mean confidence score')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # locally
    # d_rise_inference_for_evaluation(51, 101, save_result=True)
    # score_dict, iou_dict = d_rise_evaluation_result(0, 101)
    # plot_d_rise_evaluation(score_dict, iou_dict)

    # globally
    # d_rise_inference_for_evaluation_globally(0, 300, save_result=True)
    score_dict, iou_dict = d_rise_evaluation_result_globally(0, 300, iou_threshold=0.7)
    plot_d_rise_evaluation(score_dict, iou_dict)

    # compare 0.5 and 0.7 threshold
    # score_dict_05, _ = d_rise_evaluation_result_globally(0, 300, iou_threshold=0.5)
    # score_dict_07, _ = d_rise_evaluation_result_globally(0, 300, iou_threshold=0.7)
    # plot_threshold_comparison(score_dict_05, score_dict_07)