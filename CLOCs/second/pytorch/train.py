"""
    This train.py was the main function of CLOCs.
    I added some new functions.
"""


import os
import pathlib
import pickle
import shutil
import time
from functools import partial
import json
import fire
import numpy as np
import torch
from google.protobuf import text_format
from tensorboardX import SummaryWriter

import torchplus
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.utils.eval import get_coco_eval_result, get_official_eval_result, bev_box_overlap, d3_box_overlap
from second.utils.progress_bar import ProgressBar
from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import SigmoidFocalClassificationLoss
from second.pytorch.models import fusion
from second.data.preprocess import prep_pointcloud
import torchvision.transforms as transforms
from PIL import Image
from second.core.box_np_ops import remove_outside_points
# from second.kittiviewer.viewer import _riou3d_shapely
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import open3d as o3d
from scipy.spatial.transform import Rotation

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning, \
    NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


# predict_v2：主要集中在处理和生成预测结果，不涉及结果保存或评估。
# predict_kitti_to_anno：在生成预测结果后，还会进行评估和注释数据生成。
# _predict_kitti_to_file：在生成预测结果后，直接将结果保存为文件。

# Convert the values in the input dictionary to PyTorch tensors and process the data types
# according to the key names.
def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2", "d3_gt_boxes", "gt_2d_boxes"
    ]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.tensor(v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        else:
            example_torch[k] = v
    return example_torch


# By loading the configuration file and pre-trained model, initializing each component,
# and finally building and returning a 3D object detection model ready for reasoning.
def build_inference_net(config_path,
                        model_dir,
                        result_path=None,
                        predict_test=False,
                        ckpt_path=None,
                        ref_detfile=None,
                        pickle_result=True,
                        measure_time=False,
                        batch_size=1):
    model_dir = pathlib.Path(model_dir)
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
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
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    class_names = target_assigner.classes
    # 调用build，build调用了Voxelnet对net进行实例化，Voxelnet中有forward函数进行前向传播
    net = second_builder.build(model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    net.cuda()

    if ckpt_path is None:
        print("load existing model")
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)
    batch_size = batch_size or input_cfg.batch_size
    #batch_size = 1
    net.eval()
    return net


# The input data is processed by the neural network model and fusion layer to generate 3D object
# detection results and save them to a file in the format of the KITTI dataset.
def _predict_kitti_to_file(net,
                           detection_2d_path,
                           fusion_layer,
                           example,
                           result_save_path,
                           class_names,
                           center_limit_range=None,
                           lidar_input=False):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    all_3d_output_camera_dict, all_3d_output, top_predictions, fusion_input, torch_index = net(example,
                                                                                               detection_2d_path)
    t_start = time.time()
    fusion_cls_preds, flag = fusion_layer(fusion_input.cuda(), torch_index.cuda())
    t_end = time.time()
    t_fusion = t_end - t_start
    fusion_cls_preds_reshape = fusion_cls_preds.reshape(1, 200, 176, 2)
    # 可以注释掉下面这一行来运行纯SECOND
    all_3d_output.update({'cls_preds': fusion_cls_preds_reshape})
    predictions_dicts = predict_v2(net, example, all_3d_output)

    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None or preds_dict["bbox"].size.numel():
            box_2d_preds = preds_dict["bbox"].data.cpu().numpy()
            box_preds = preds_dict["box3d_camera"].data.cpu().numpy()
            scores = preds_dict["scores"].data.cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].data.cpu().numpy()
            # write pred to file
            box_preds = box_preds[:, [0, 1, 2, 4, 5, 3,
                                      6]]  # lhw->hwl(label file format)
            label_preds = preds_dict["label_preds"].data.cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            result_lines = []
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                result_dict = {
                    'name': class_names[int(label)],
                    'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                    'bbox': bbox,
                    'location': box[:3],
                    'dimensions': box[3:6],
                    'rotation_y': box[6],
                    'score': score,
                }
                result_line = kitti.kitti_result_line(result_dict)
                result_lines.append(result_line)
        else:
            result_lines = []
        result_file = f"{result_save_path}/{kitti.get_image_index_str(img_idx)}.txt"
        result_str = '\n'.join(result_lines)
        with open(result_file, 'w') as f:
            f.write(result_str)


# Generate annotation data in KITTI format from the model's prediction results. This function generates 3D detection
# results through forward propagation, calculates classification loss, and converts the final detection results into
# KITTI format annotations.
# Added parameter flag_2d. When true, directly accept the file path of 2d detection results.
# The default is false.
def predict_kitti_to_anno(net,
                          detection_2d_path,
                          fusion_layer,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False,
                          global_set=None,
                          flag_2d=False):
    focal_loss_val = SigmoidFocalClassificationLoss()
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']

    # 模型推理，前向传播，返回5个结果 forward函数位于pytorch/models/voxelnet.py
    # 相比于原本的second，多出了detection_2d_path
    # all_3d_output_camera_dict: 包含3D检测结果的字典，可能是相机坐标系中的结果。
    # all_3d_output: 包含3D检测结果的字典，可能是激光雷达坐标系中的结果。
    # top_predictions: 顶部的预测结果，可能是2D或3D检测的预测框和类别。
    # fusion_input: 用于多模态融合层的输入特征。
    # torch_index: 用于索引的张量，可能是用于多模态融合的索引信息。
    (all_3d_output_camera_dict,
     all_3d_output,
     top_predictions,
     fusion_input,
     torch_index) = net(example, detection_2d_path, flag_2d)
    # print("all_3d_output_camera_dict: ", all_3d_output_camera_dict)
    # print("all_3d_output: ", all_3d_output)
    # print("top_predictions: ", top_predictions)
    print("fusion_input: ", fusion_input)
    print("torch_index: ", torch_index)
    # save_path = '/home/xkx/桌面/fusion_input.pt'
    # torch.save(fusion_input, save_path)
    # save_path = '/home/xkx/桌面/torch_index.pt'
    # torch.save(torch_index, save_path)


    t_start = time.time()
    # fusion inference, forward function is in pytorch/models/fusion.py
    fusion_cls_preds, flag = fusion_layer(fusion_input.cuda(), torch_index.cuda())
    print(f'fusion_cls_preds: {fusion_cls_preds}')
    print(f'flag: {flag}')
    t_end = time.time()
    t_fusion = t_end - t_start
    fusion_cls_preds_reshape = fusion_cls_preds.reshape(1, 200, 176, 2)
    all_3d_output.update({'cls_preds': fusion_cls_preds_reshape})
    predictions_dicts = predict_v2(net, example, all_3d_output)

    test_mode = False
    if test_mode == False and 'd3_gt_boxes' in example:
        d3_gt_boxes = example["d3_gt_boxes"][0, :, :]
        if d3_gt_boxes.shape[0] == 0:
            target_for_fusion = np.zeros((1, 70400, 1))
            positives = torch.zeros(1, 70400).type(torch.float32).cuda()
            negatives = torch.zeros(1, 70400).type(torch.float32).cuda()
            negatives[:, :] = 1
        else:
            d3_gt_boxes_camera = box_torch_ops.box_lidar_to_camera(
                d3_gt_boxes, example['rect'][0, :], example['Trv2c'][0, :])
            d3_gt_boxes_camera_bev = d3_gt_boxes_camera[:, [0, 2, 3, 5, 6]]
            ###### predicted bev boxes
            pred_3d_box = all_3d_output_camera_dict[0]["box3d_camera"]
            pred_bev_box = pred_3d_box[:, [0, 2, 3, 5, 6]]
            #iou_bev = bev_box_overlap(d3_gt_boxes_camera_bev.detach().cpu().numpy(), pred_bev_box.detach().cpu().numpy(), criterion=-1)
            iou_bev = d3_box_overlap(d3_gt_boxes_camera.detach().cpu().numpy(),
                                     pred_3d_box.squeeze().detach().cpu().numpy(), criterion=-1)
            iou_bev_max = np.amax(iou_bev, axis=0)
            target_for_fusion = ((iou_bev_max >= 0.7) * 1).reshape(1, -1, 1)
            positive_index = ((iou_bev_max >= 0.7) * 1).reshape(1, -1)
            positives = torch.from_numpy(positive_index).type(torch.float32).cuda()
            negative_index = ((iou_bev_max <= 0.5) * 1).reshape(1, -1)
            negatives = torch.from_numpy(negative_index).type(torch.float32).cuda()

        cls_preds = fusion_cls_preds
        one_hot_targets = torch.from_numpy(target_for_fusion).type(torch.float32).cuda()

        negative_cls_weights = negatives.type(torch.float32) * 1.0
        cls_weights = negative_cls_weights + 1.0 * positives.type(torch.float32)
        pos_normalizer = positives.sum(1, keepdim=True).type(torch.float32)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_losses = focal_loss_val._compute_loss(cls_preds, one_hot_targets, cls_weights.cuda())  # [N, M]

        cls_losses_reduced = cls_losses.sum() / example['labels'].shape[0]
        cls_losses_reduced = cls_losses_reduced.detach().cpu().numpy()
    else:
        cls_losses_reduced = 1000
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None or preds_dict["bbox"].size.numel() != 0:
            box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
            box_preds = preds_dict["box3d_camera"].detach().cpu().numpy()
            scores = preds_dict["scores"].detach().cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
            # write pred to file
            label_preds = preds_dict["label_preds"].detach().cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            anno = kitti.get_start_result_anno()
            num_example = 0
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                     box[6])
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array(
            [img_idx] * num_example, dtype=np.int64)
        #cls_losses_reduced=100
    return annos, cls_losses_reduced, predictions_dicts


# evaluate the accuracy on KITTI evaluation set
def evaluate(config_path='./configs/car.fhd.config',
             second_model_dir='/home/xkx/CLOCs/model_dir/second_model',
             fusion_model_dir='../CLOCs_SecCas_pretrained',
             result_path='/home/xkx/CLOCs/eval_results',
             predict_test=False,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=True,
             measure_time=False,
             batch_size=1):
    second_model_dir = pathlib.Path(second_model_dir)
    print("Predict_test: ", predict_test)
    if predict_test:
        result_name = 'predict_test'
    else:
        result_name = 'eval_results'
    # 如果没有提供存放结果的路径的话，将结果保存在model_dir下
    if result_path is None:
        result_path = second_model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    detection_2d_path = config.train_config.detection_2d_path
    center_limit_range = model_cfg.post_center_limit_range
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    # 输出sparse_shape: [  41 1600 1408]，表示体素的形状
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    # 输出num_class is : 1，表示检测的目标类别
    class_names = target_assigner.classes
    # this one is used for training car detector
    # 输出load existing model
    net = build_inference_net(config_path, second_model_dir)
    fusion_layer = fusion.fusion()
    fusion_layer.cuda()
    net.cuda()

    ############ restore parameters for fusion layer
    if ckpt_path is None:
        print("load existing model for fusion layer")
        # 输出Restoring parameters from ../CLOCs_SecCas_pretrained/fusion_layer-11136.tckpt
        torchplus.train.try_restore_latest_checkpoints(fusion_model_dir, [fusion_layer])
    else:
        torchplus.train.restore(ckpt_path, fusion_layer)
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    batch_size = batch_size or input_cfg.batch_size
    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=not predict_test,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    fusion_layer.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    dt_annos = []
    global_set = None
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)
    prep_example_times = []
    prep_times = []
    t2 = time.time()
    val_loss_final = 0
    for example in iter(eval_dataloader):
        if measure_time:
            prep_times.append(time.time() - t2)
            t1 = time.time()
            torch.cuda.synchronize()
        example = example_convert_to_torch(example, float_dtype)
        if measure_time:
            torch.cuda.synchronize()
            prep_example_times.append(time.time() - t1)

        if pickle_result:
            with torch.no_grad():
                dt_annos_i, val_losses, _ = predict_kitti_to_anno(
                    net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                    model_cfg.lidar_input, global_set)
            dt_annos += dt_annos_i
            val_loss_final = val_loss_final + val_losses
        else:
            _predict_kitti_to_file(net, detection_2d_path, fusion_layer, example, result_path_step, class_names,
                                   center_limit_range, model_cfg.lidar_input)
        bar.print_bar()
        if measure_time:
            t2 = time.time()

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')
    print("validation_loss:", val_loss_final / len(eval_dataloader))
    if measure_time:
        print(f"avg example to torch time: {np.mean(prep_example_times) * 1000:.3f} ms")
        print(f"avg prep time: {np.mean(prep_times) * 1000:.3f} ms")
    for name, val in net.get_avg_time_dict().items():
        print(f"avg {name} time = {val * 1000:.3f} ms")
    if not predict_test:
        gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
        if not pickle_result:
            dt_annos = kitti.get_label_annos(result_path_step)
        result = get_official_eval_result(gt_annos, dt_annos, class_names)
        # print(json.dumps(result, indent=2))
        print(result)
        # get_coco这里还有一点bug但是应该影响不大
        # result = get_coco_eval_result(gt_annos, dt_annos, class_names)
        # print(result)
        if pickle_result:
            with open(result_path_step / "result.pkl", 'wb') as f:
                pickle.dump(dt_annos, f)
    else:
        if pickle_result:
            with open(result_path_step / "result.pkl", 'wb') as f:
                pickle.dump(dt_annos, f)


def save_config(config_path, save_path):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    ret = text_format.MessageToString(config, indent=2)
    with open(save_path, 'w') as f:
        f.write(ret)


# Post-processing, receiving a large number of proposals in preds_dict, and returning the final detection results
def predict_v2(net, example, preds_dict):
    # keys in preds_dict:,  dict_keys(['box_preds', 'cls_preds', 'dir_cls_preds'])
    t = time.time()
    batch_size = example['anchors'].shape[0]
    batch_anchors = example["anchors"].view(batch_size, -1, 7)
    batch_rect = example["rect"]
    batch_Trv2c = example["Trv2c"]
    batch_P2 = example["P2"]
    if "anchors_mask" not in example:
        batch_anchors_mask = [None] * batch_size
    else:
        batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
    batch_imgidx = example['image_idx']

    t = time.time()
    batch_box_preds = preds_dict["box_preds"]
    batch_cls_preds = preds_dict["cls_preds"]
    batch_box_preds = batch_box_preds.view(batch_size, -1,
                                           net._box_coder.code_size)
    num_class_with_bg = net._num_class
    if not net._encode_background_as_zeros:
        num_class_with_bg = net._num_class + 1
    batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                           num_class_with_bg)
    batch_box_preds = net._box_coder.decode_torch(batch_box_preds,
                                                  batch_anchors)
    if net._use_direction_classifier:
        batch_dir_preds = preds_dict["dir_cls_preds"]
        batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
    else:
        batch_dir_preds = [None] * batch_size

    predictions_dicts = []
    for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
            batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
            batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask):
        if a_mask is not None:
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
        box_preds = box_preds.float()
        cls_preds = cls_preds.float()
        rect = rect.float()
        Trv2c = Trv2c.float()
        P2 = P2.float()
        if net._use_direction_classifier:
            if a_mask is not None:
                dir_preds = dir_preds[a_mask]
            dir_labels = torch.max(dir_preds, dim=-1)[1]
        if net._encode_background_as_zeros:
            # this don't support softmax
            assert net._use_sigmoid_score is True
            total_scores = torch.sigmoid(cls_preds)
        else:
            # encode background as first element in one-hot vector
            if net._use_sigmoid_score:
                total_scores = torch.sigmoid(cls_preds)[..., 1:]
            else:
                total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
        # Apply NMS in birdeye view
        if net._use_rotate_nms:
            nms_func = box_torch_ops.rotate_nms
        else:
            nms_func = box_torch_ops.nms

        if net._multiclass_nms:
            # curently only support class-agnostic boxes.
            boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
            if not net._use_rotate_nms:
                box_preds_corners = box_torch_ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                    boxes_for_nms[:, 4])
                boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                    box_preds_corners)
            boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
            selected_per_class = box_torch_ops.multiclass_nms(
                nms_func=nms_func,
                boxes=boxes_for_mcnms,
                scores=total_scores,
                num_class=net._num_class,
                pre_max_size=net._nms_pre_max_size,
                post_max_size=net._nms_post_max_size,
                iou_threshold=net._nms_iou_threshold,
                score_thresh=net._nms_score_threshold,
            )
            selected_boxes, selected_labels, selected_scores = [], [], []
            selected_dir_labels = []
            for i, selected in enumerate(selected_per_class):
                if selected is not None:
                    num_dets = selected.shape[0]
                    selected_boxes.append(box_preds[selected])
                    selected_labels.append(
                        torch.full([num_dets], i, dtype=torch.int64))
                    if net._use_direction_classifier:
                        selected_dir_labels.append(dir_labels[selected])
                    selected_scores.append(total_scores[selected, i])
            selected_boxes = torch.cat(selected_boxes, dim=0)
            selected_labels = torch.cat(selected_labels, dim=0)
            selected_scores = torch.cat(selected_scores, dim=0)
            if net._use_direction_classifier:
                selected_dir_labels = torch.cat(
                    selected_dir_labels, dim=0)
        else:
            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(
                    total_scores.shape[0],
                    device=total_scores.device,
                    dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(total_scores, dim=-1)

            if net._nms_score_threshold > 0.0:
                thresh = torch.tensor(
                    [net._nms_score_threshold],
                    device=total_scores.device).type_as(total_scores)
                top_scores_keep = (top_scores >= thresh)
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if net._nms_score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    if net._use_direction_classifier:
                        dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not net._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)
                # the nms in 3d detection just remove overlap boxes.
                selected = nms_func(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=net._nms_pre_max_size,
                    post_max_size=net._nms_post_max_size,
                    iou_threshold=net._nms_iou_threshold,
                )

            else:
                selected = []
            # if selected is not None:
            selected_boxes = box_preds[selected]
            if net._use_direction_classifier:
                selected_dir_labels = dir_labels[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

        # finally generate predictions.
        if selected_boxes.shape[0] != 0:
            box_preds = selected_boxes
            scores = selected_scores
            label_preds = selected_labels
            if net._use_direction_classifier:
                dir_labels = selected_dir_labels
                #print("dir_labels shape is:",dir_labels.shape,"the values are: ",dir_labels)
                opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                box_preds[..., -1] += torch.where(
                    opp_labels,
                    torch.tensor(np.pi).type_as(box_preds),
                    torch.tensor(0.0).type_as(box_preds))
            # final_box_preds is detection result in 3d lidar coordinate
            final_box_preds = box_preds
            final_scores = scores
            final_labels = label_preds
            final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                final_box_preds, rect, Trv2c)
            locs = final_box_preds_camera[:, :3]
            dims = final_box_preds_camera[:, 3:6]
            angles = final_box_preds_camera[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            box_corners = box_torch_ops.center_to_corner_box3d(
                locs, dims, angles, camera_box_origin, axis=1)

            box_corners_in_image = box_torch_ops.project_to_image(
                box_corners, P2)
            # box_corners_in_image: [N, 8, 2]
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            # Steps to get the 2d bbox:
            # 1. bbox in 3d lidar coordinate from NN
            # 2. convert the bbox in 3d lidar into 3d camera coordinate and calculate the 8 corners
            # 3. project corners to 2d pixel coordinate
            # 4. find the min and max values of these 8 corners
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            # predictions
            predictions_dict = {
                "bbox": box_2d_preds,
                "box3d_camera": final_box_preds_camera,
                "box3d_lidar": final_box_preds,
                "scores": final_scores,
                "label_preds": label_preds,
                "image_idx": img_idx,
            }
        else:
            dtype = batch_box_preds.dtype
            device = batch_box_preds.device
            predictions_dict = {
                "bbox": torch.zeros([0, 4], dtype=dtype, device=device),
                "box3d_camera": torch.zeros([0, 7], dtype=dtype, device=device),
                "box3d_lidar": torch.zeros([0, 7], dtype=dtype, device=device),
                "scores": torch.zeros([0], dtype=dtype, device=device),
                "label_preds": torch.zeros([0, 4], dtype=top_labels.dtype, device=device),
                "image_idx": img_idx,
            }
        predictions_dicts.append(predictions_dict)
    # keys in predictions_dict:, dict_keys(['bbox', 'box3d_camera', 'box3d_lidar', 'scores', 'label_preds',
    # 'image_idx'])
    return predictions_dicts


def save_example(config_path='../configs/car.fhd.config'):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    detection_2d_path = config.train_config.detection_2d_path
    center_limit_range = model_cfg.post_center_limit_range
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    # 输出sparse_shape: [  41 1600 1408]，表示体素的形状
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    # 输出num_class is : 1，表示检测的目标类别
    class_names = target_assigner.classes

    batch_size = input_cfg.batch_size
    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    save_path = '/home/xkx/kitti/example'
    for example in iter(eval_dataloader):
        example = example_convert_to_torch(example, float_dtype)
        first_value = example['image_idx'][0]
        file_name = f'{first_value:06}.pkl'
        full_path = os.path.join(save_path, file_name)
        print(full_path)
        with open(full_path, 'wb') as f:
            pickle.dump(example, f)
        print(f'Dictionary saved as {full_path}')


# inference with CLOCs late fusion model
# run 3001 times
# one time for original, namely no perturbation
# 3000 times for masked point cloud
def inference_original_and_masked_input(start_idx, end_idx, save_result=False, it_nr=3000, batch_size=8,
                                        config_path='/home/xkx/CLOCs/second/configs/car.fhd.config',
                                        second_model_dir='../model_dir/second_model',
                                        fusion_model_dir='../CLOCs_SecCas_pretrained'
                                        ):
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
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
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
        input_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_croped_by_occam/{idx_str}.bin'
        i_path = '/home/xkx/kitti/training/image_2/' + idx_str + '.png'

        info = read_kitti_info_val(idx=idx)
        input_pc = np.fromfile(input_path, dtype=np.float32)
        input_pc = input_pc.reshape(-1, 4)
        print(f"input_pc.shape: {input_pc.shape}")

        example_original = get_inference_input_dict(config=config,
                                                    voxel_generator=voxel_generator,
                                                    target_assigner=target_assigner,
                                                    info=info,
                                                    points=input_pc,
                                                    i_path=i_path)

        example_original = example_convert_to_torch(example_original, torch.float32)

        with torch.no_grad():
            dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                net, detection_2d_path, fusion_layer, example_original, class_names, center_limit_range,
                model_cfg.lidar_input)
            prediction_dicts = prediction_dicts[0]
            print("original prediction result: ", prediction_dicts)

        if save_result:
            save_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_original_dt_results/{idx_str}_original.pkl'
            with open(save_path, 'wb') as output_file:
                pickle.dump(prediction_dicts, output_file)
            print(f"detection results of original {idx_str}.bin have been saved")

        results = []
        num = 0
        # read masked input from occam
        masked_input_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_masked_pointcloud_2/{idx_str}_{it_nr}.pkl'

        with open(masked_input_path, 'rb') as file:
            data = pickle.load(file)
        for i in range(it_nr):
            j, k = i // batch_size, i % batch_size
            batch_masked_input_pc = data[j]['points']
            masked_input_pc = batch_masked_input_pc[batch_masked_input_pc[:, 0] == k]
            masked_input_pc = masked_input_pc[:, 1:]
            # print(masked_input_pc.shape)
            example = get_inference_input_dict(config=config,
                                               voxel_generator=voxel_generator,
                                               target_assigner=target_assigner,
                                               info=info,
                                               points=masked_input_pc,
                                               i_path=i_path)

            example = example_convert_to_torch(example, torch.float32)

            with torch.no_grad():
                dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                    net, detection_2d_path, fusion_layer, example, class_names, center_limit_range,
                    model_cfg.lidar_input)
                prediction_dicts = prediction_dicts[0]
                # print("prediction_dicts: ", prediction_dicts)
            results.append(prediction_dicts)
        if save_result:
            save_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_masked_dt_results_2/{idx_str}_{it_nr}.pkl'
            with open(save_path, 'wb') as output_file:
                pickle.dump(results, output_file)
            print(f"detection results of masked {idx_str}.bin have been saved")


def get_inference_input_dict(config, voxel_generator, target_assigner, info, points, i_path):
    rect = info['calib/R0_rect']
    P2 = info['calib/P2']
    Trv2c = info['calib/Tr_velo_to_cam']
    input_cfg = config.eval_input_reader
    model_cfg = config.model.second

    pil2tensor = transforms.ToTensor()
    pil_image = Image.open(str(i_path)).convert("RGB")
    image_pang_bgr = np.array(pil_image)[:, :, [2, 1, 0]]
    image_pang = image_pang_bgr
    '''
    pil2tensor = transforms.ToTensor()
    pil_image = Image.open(str(i_path))
    image_pang = pil2tensor(pil_image)
    '''
    input_dict = {
        'points': points,
        'rect': rect,
        'Trv2c': Trv2c,
        'P2': P2,
        'image_shape': np.array(info["img_shape"], dtype=np.int32),
        'image_idx': info['image_idx'],
        'image_path': info['img_path'],
        'images': image_pang
        # 'pointcloud_num_features': num_point_features,
    }
    # out_size_factor = model_cfg.rpn.layer_strides[0] // model_cfg.rpn.upsample_strides[0]

    example = prep_pointcloud(
        input_dict=input_dict,
        root_path=None,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        max_voxels=input_cfg.max_number_of_voxels,
        class_names=target_assigner.classes,
        training=False,
        create_targets=False,
        shuffle_points=input_cfg.shuffle_points,
        generate_bev=False,
        without_reflectivity=model_cfg.without_reflectivity,
        num_point_features=model_cfg.num_point_features,
        anchor_area_threshold=input_cfg.anchor_area_threshold,
        anchor_cache=None,
        out_size_factor=8,
        out_dtype=np.float32)
    example["image_idx"] = info['image_idx']
    example["image_shape"] = input_dict["image_shape"]
    example["points"] = points
    if "anchors_mask" in example:
        example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
    #############
    # convert example to batched example
    #############
    example = merge_second_batch([example])
    return example


def read_kitti_info_val(idx):
    file_path = "/home/xkx/kitti/kitti_infos_trainval.pkl"
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    for item in data:
        if item.get('image_idx') == idx:
            return item
    return IndexError


def inference_original_input(start_idx, end_idx, save_result=False,
                             config_path='/home/xkx/CLOCs/second/configs/car.fhd.config',
                             second_model_dir='../model_dir/second_model',
                             fusion_model_dir='../CLOCs_SecCas_pretrained'
                             ):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    # detection_2d_path = config.train_config.detection_2d_path
    model_cfg = config.model.second
    center_limit_range = model_cfg.post_center_limit_range
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
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
        input_path = f'/home/xkx/kitti/training/velodyne/{idx_str}.bin'
        i_path = '/home/xkx/kitti/training/image_2/' + idx_str + '.png'

        info = read_kitti_info_val(idx=idx)
        input_pc = np.fromfile(input_path, dtype=np.float32)
        input_pc = input_pc.reshape(-1, 4)
        print(f"input_pc.shape: {input_pc.shape}")

        example_original = get_inference_input_dict(config=config,
                                                    voxel_generator=voxel_generator,
                                                    target_assigner=target_assigner,
                                                    info=info,
                                                    points=input_pc,
                                                    i_path=i_path)

        example_original = example_convert_to_torch(example_original, torch.float32)

        detection_2d_path = f'/home/xkx/CLOCs/d2_detection_data/{idx_str}.txt'
        print(f"detection 2d: {detection_2d_path}")
        with torch.no_grad():
            dt_annos, val_losses, prediction_dicts = predict_kitti_to_anno(
                net, detection_2d_path, fusion_layer, example_original, class_names, center_limit_range,
                model_cfg.lidar_input, flag_2d=True)
            prediction_dicts = prediction_dicts[0]
            print("original prediction result: ", prediction_dicts)

        if save_result:
            save_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_original_dt_results/{idx_str}_original.pkl'
            with open(save_path, 'wb') as output_file:
                pickle.dump(prediction_dicts, output_file)
            print(f"detection results of original {idx_str}.bin have been saved")


if __name__ == '__main__':
    # fire.Fire()
    inference_original_input(2, 3)
    # inference_original_and_masked_input(1, 2, save_result=False)