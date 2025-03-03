"""
    This file generates 2D heat maps.
"""

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

from train import (build_inference_net,
                   example_convert_to_torch,
                   get_inference_input_dict,
                   predict_kitti_to_anno)


def mask_image(image, mask):
    masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
              255).astype(np.uint8)
    return masked


# inference with unmasked 3d and masked 2d
def d_rise(start_idx, end_idx, it_nr=3000, save_result=False,
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

    # inference with unchanged 3D pipeline and perturbed 2D image
    for idx in range(start_idx, end_idx):
        idx_str = str(idx).zfill(6)
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

        results = []

        # 3000 iterations
        for i in range(it_nr):
            # Perturbed 2D detection results
            detection_2d_path = (f"/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/"
                                 f"CasCade_2d_detection_results_3000/{idx_str}/results/{i}.txt")

            with torch.no_grad():
                dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                    net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                    model_cfg.lidar_input, flag_2d=True)
                prediction_dicts = prediction_dicts[0]
                # print(f"pred: {prediction_dicts}")
                results.append(prediction_dicts)

        if save_result:
            save_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CLOC_2d_detection_results/'
                         f'{idx_str}_{it_nr}.pkl')
            with open(save_path, 'wb') as output_file:
                pickle.dump(results, output_file)
            print(f"2d detection result of {it_nr} masked {idx_str}.png have been saved")


# deprecated
def inference_original_2d_with_self_CasCade(start_idx, end_idx, save_result=False,
           config_path='/home/xkx/CLOCs/second/configs/car.fhd.config',
           second_model_dir='../model_dir/second_model',
           fusion_model_dir='../CLOCs_SecCas_pretrained'):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

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

        detection_2d_path = (f"/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/"
                             f"CasCade_2d_detection_results_original/{idx_str}.txt")

        with torch.no_grad():
            dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                model_cfg.lidar_input, flag_2d=True)
            prediction_dicts = prediction_dicts[0]
            print(f"pred: {prediction_dicts}")

        if save_result:
            save_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CLOC_2d_detection_results_original/'
                         f'{idx_str}.pkl')
            with open(save_path, 'wb') as output_file:
                pickle.dump(prediction_dicts, output_file)
            print(f"2d detection result of original {idx_str}.png have been saved")


def read_kitti_info_val(idx):
    file_path = "/home/xkx/kitti/kitti_infos_trainval.pkl"
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    for item in data:
        if item.get('image_idx') == idx:
            return item
    return IndexError


def read_original_dt_results_2d(idx_str):
    read_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_original_dt_results/'
                 f'{idx_str}_original.pkl')
    # read_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CLOC_2d_detection_results_original/'
    #              f'{idx_str}.pkl')
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


def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_cam(image, mask):
    """
    generate CAM image
    :param image: [H,W,C], raw image
    :param mask: [H,W], 0~1
    :return: tuple(cam,heatmap)
    """
    # mask to heatmap
    mask = norm_image(mask)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # heatmap = heatmap[..., ::-1]

    # merge heatmap to original image
    cam = 0.5 * heatmap + 0.5 * image
    return norm_image(cam), heatmap


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


def generate_saliency_map(idx, n_masks=3000,
                          save_figures=False,
                          save_heatmap=False,
                          show=False,
                          save_important_masks=False):
    # base detection and number of objects
    base_det_boxes, base_det_labels, base_det_scores = read_original_dt_results_2d(str(idx).zfill(6))
    num_of_objects = base_det_boxes.shape[0]
    if num_of_objects == 0:
        heat_map = np.zeros((0, 0, 0))
    else:
        # read image
        i_path = f'/home/xkx/kitti/training/image_2/{str(idx).zfill(6)}.png'
        image = cv2.imread(i_path)
        image_h, image_w = image.shape[:2]

        # read masks
        mask_path = (f'//media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CasCade_2d_detection_results_3000'
                     f'/{str(idx).zfill(6)}/masks/masks_{n_masks}.pkl')
        with open(mask_path, 'rb') as file:
            masks = pickle.load(file)

        # read 2D detection results
        CLOC_2d_detection = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CLOC_2d_detection_results'
                             f'/{str(idx).zfill(6)}_{n_masks}.pkl')
        with open(CLOC_2d_detection, 'rb') as file:
            preds = pickle.load(file)

        # Initialize heat map
        heat_map = np.zeros((num_of_objects, image_h, image_w), dtype=np.float32)
        mask_indexes = np.zeros((num_of_objects, n_masks), dtype=np.int32)

        # Generate heat map for each detected object in the image
        for target_index in range(num_of_objects):
            target_box = base_det_boxes[target_index]
            print(f"target_box: {target_box}")

            scores = np.zeros(n_masks)
            ious = np.zeros(n_masks)
            for i in range(n_masks):
                mask = masks[i]
                pred = preds[i]
                pred_boxes = pred["bbox"]
                pred_scores = pred["scores"]
                pred_boxes = pred_boxes.cpu().numpy()
                pred_scores = pred_scores.cpu().numpy()

                score = 0
                best_score_value = None
                for box, score_value in zip(pred_boxes, pred_scores):
                    current_score = iou(target_box, box) * score_value
                    if current_score > score:
                        score = current_score
                        scores[i] = current_score
                        ious[i] = current_score / score_value
                        best_score_value = score_value
                heat_map[target_index] += mask * score

            print(f'iou.mean: {np.mean(ious)}')
            print(f'iou.max:{np.max(ious)}')
            print(f'iou.min:{np.min(ious)}')

            # Sort the mask indexes by the scores
            mask_indexes[target_index] = np.argsort(scores)

            image_with_bbox, _ = gen_cam(image, heat_map[target_index])
            left_corner = target_box[:2].astype(np.int32)
            right_corner = target_box[2:].astype(np.int32)
            cv2.rectangle(image_with_bbox, left_corner, right_corner,
                          (0, 0, 255), 5)
            if save_figures:
                save_path = (f"/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/D_RISE_heat_map_figures_2/"
                             f"{str(idx).zfill(6)}_{target_index}_{n_masks}.png")
                cv2.imwrite(save_path, image_with_bbox)
                print(f"{str(idx).zfill(6)}_{target_index}_{n_masks}.png has been saved")

            if show:
                cv2.imshow('image', image_with_bbox)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        if save_important_masks:
            save_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/D_RISE_heat_map_important_mask_indexes_2/'
                         f'{str(idx).zfill(6)}_30.pkl')
            with open(save_path, 'wb') as output_file:
                pickle.dump(mask_indexes, output_file)

    if save_heatmap:
        save_path = (f"/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/D_RISE_heat_map_data_2/"
                     f"{str(idx).zfill(6)}_{n_masks}.pkl")
        with open(save_path, 'wb') as output_file:
            pickle.dump(heat_map, output_file)
        print(f"heat map has been saved: {heat_map}")
    return heat_map


# only for debugging
def visualize_preds(idx, index_of_mask, n_masks=3000):
    base_det_boxes, base_det_labels, base_det_scores = read_original_dt_results_2d(str(idx).zfill(6))
    print(base_det_boxes)
    num_of_objects = base_det_boxes.shape[0]
    i_path = f'/home/xkx/kitti/training/image_2/{str(idx).zfill(6)}.png'
    image = cv2.imread(i_path)
    image_h, image_w = image.shape[:2]
    mask_path = (f'//media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CasCade_2d_detection_results_3000'
                 f'/{str(idx).zfill(6)}/masks/masks_{n_masks}.pkl')
    with open(mask_path, 'rb') as file:
        masks = pickle.load(file)
    CLOC_2d_detection = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CLOC_2d_detection_results'
                         f'/{str(idx).zfill(6)}_{n_masks}.pkl')
    with open(CLOC_2d_detection, 'rb') as file:
        preds = pickle.load(file)

    for i in range(n_masks):
        if i in index_of_mask:
            print(i)
            mask = masks[i]
            masked_image = image.copy()
            masked_image = mask_image(masked_image, mask)
            pred = preds[i]
            pred_boxes = pred["bbox"]
            pred_scores = pred["scores"]
            pred_boxes = pred_boxes.cpu().numpy()
            pred_scores = pred_scores.cpu().numpy()
        # Generate heat map for each detected object in the image
        #     for pred_box in pred_boxes:
        #         left_corner = pred_box[:2].astype(np.int32)
        #         right_corner = pred_box[2:].astype(np.int32)
        #         cv2.rectangle(masked_image, left_corner, right_corner,
        #                       (0, 0, 255), 5)
            left_corner = base_det_boxes[0][:2].astype(np.int32)
            right_corner = base_det_boxes[0][2:].astype(np.int32)
            cv2.rectangle(masked_image, left_corner, right_corner,
                          (0, 0, 255), 5)

            cv2.imshow('image', masked_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    # inference with unmasked 3d and masked 2d
    # d_rise(214, 301, save_result=True)

    for i in range(10, 11):
        generate_saliency_map(i, save_figures=False, save_heatmap=False, show=True, save_important_masks=False)

    # visualize the process of masked inference
    # index = np.arange(0, 3000)
    # visualize_preds(10, index)
    # idx = 10
    # base_det_boxes, base_det_labels, base_det_scores = read_original_dt_results_2d(str(idx).zfill(6))
    # print(base_det_boxes)
