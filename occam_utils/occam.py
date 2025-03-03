"""
    This class can be divided into three parts, they are separated by dividers.
    The first part is original codes from OccAM authors.
    The second part is implemented by Kexuan for his Master's thesis. This part focus on adapting the code to enable it for the fusion.
    The third part is implemented by Kexuan as well, but for integrating CLOCs into OccAM.
"""

import copy
import torch
import open3d
import tqdm
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
from mmdet.apis import DetInferencer
import mmcv

from torch.utils.data import DataLoader
import torch.nn.functional as F
from CLOCs.fusion_network.fusion import fusion
import torchplus

from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from scipy.spatial.transform import Rotation

from occam_utils.occam_datasets import BaseDataset, OccamInferenceDataset
from occam_utils.utils import (decode_torch, build_stage2_training, generate_anchors,
                               read_kitti_info_val, example_convert_to_torch,
                               merge_second_batch)
from occam_utils import utils

from second.protos import pipeline_pb2
from google.protobuf import text_format

import torchvision.transforms as transforms
from PIL import Image


class OccAM(object):
    """
    OccAM base class to store model, cfg and offer operations to preprocess the
    data and compute the attribution maps
    """

    def __init__(self, data_config, model_config, occam_config, class_names,
                 model_ckpt_path, nr_it, logger):
        """
        Parameters
        ----------
            data_config : EasyDict
               dataset cfg including data preprocessing properties (OpenPCDet)
            model_config : EasyDict
               object detection model definition (OpenPCDet)
            occam_config: EasyDict
                sampling properties for attribution map generation, see cfg file
            class_names :
                list of class names (OpenPCDet)
            model_ckpt_path: str
                path to pretrained model weights
            nr_it : int
                number of sub-sampling iterations; the higher, the more accurate
                are the resulting attribution maps
            logger: Logger
        """
        self.data_config = data_config
        self.model_config = model_config
        self.occam_config = occam_config
        self.class_names = class_names
        self.logger = logger
        self.nr_it = nr_it

        self.base_dataset = BaseDataset(data_config=self.data_config,
                                        class_names=self.class_names,
                                        occam_config=self.occam_config)

        # 创建推理模型的网络，用的是pcdet的api
        self.model = build_network(model_cfg=self.model_config,
                                   num_class=len(self.class_names),
                                   dataset=self.base_dataset)
        # 从ckpt中读取预训练的模型参数
        self.model.load_params_from_file(filename=model_ckpt_path,
                                         logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

        def read_CLOCs_configs(config_path='CLOCs/second/configs/car.fhd.config'):
            config = pipeline_pb2.TrainEvalPipelineConfig()
            with open(config_path, "r") as f:
                proto_str = f.read()
                text_format.Merge(proto_str, config)

            model_cfg = config.model.second

            return model_cfg

        # add CLOCs configs
        self.CLOCs_cfg = read_CLOCs_configs()
        self._encode_background_as_zeros = True
        self._use_direction_classifier = True
        self._use_sigmoid_score = True
        self.code_size = 7
        self._num_class = 1
        self.anchor_ranges = [0, -40.0, -1.78, 70.4, 40.0, -1.78]
        self.target_class = 'car'

    def load_and_preprocess_pcl(self, source_file_path):
        """
        load given point cloud file and preprocess data according OpenPCDet
        data config using the base dataset

        Parameters
        ----------
        source_file_path : str
            path to point cloud to analyze (bin or npy)

        Returns
        -------
        pcl : ndarray (N, 4)
            preprocessed point cloud (x, y, z, intensity)
        """
        pcl = self.base_dataset.load_and_preprocess_pcl(source_file_path)
        return pcl

    def get_base_predictions(self, pcl):
        """
        get all K detections in full point cloud for which attribution maps will
        be determined

        Parameters
        ----------
        pcl : ndarray (N, 4)
            preprocessed point cloud (x, y, z, intensity)

        Returns
        -------
        base_det_boxes : ndarray (K, 7)
            bounding box parameters of detected objects
        base_det_labels : ndarray (K)
            labels of detected objects
        base_det_scores : ndarray (K)
            confidence scores for detected objects
        """
        input_dict = {
            'points': pcl
        }

        data_dict = self.base_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.base_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        # 模型推理
        with torch.no_grad():
            base_pred_dict, _ = self.model.forward(data_dict)

        base_det_boxes = base_pred_dict[0]['pred_boxes'].cpu().numpy()
        base_det_labels = base_pred_dict[0]['pred_labels'].cpu().numpy()
        base_det_scores = base_pred_dict[0]['pred_scores'].cpu().numpy()

        return base_det_boxes, base_det_labels, base_det_scores

    def merge_detections_in_batch(self, det_dicts):
        """
        In order to efficiently determine the confidence score for
        all detections in a batch they are merged.

        Parameters
        ----------
        det_dicts : list
            list of M dicts containing the detections in the M samples within
            the batch (pred boxes, pred scores, pred labels)

        Returns
        -------
        pert_det_boxes : ndarray (L, 7)
            bounding boxes of all L detections in the M samples
        pert_det_labels : ndarray (L)
            labels of all L detections in the M samples
        pert_det_scores : ndarray (L)
            scores of all L detections in the M samples
        batch_ids : ndarray (L)
            Mapping of the detections to the individual samples within the batch
        """
        batch_ids = []

        data_dict = defaultdict(list)
        for batch_id, cur_sample in enumerate(det_dicts):
            batch_ids.append(
                np.ones(cur_sample['pred_labels'].shape[0], dtype=int)
                * batch_id)

            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_ids = np.concatenate(batch_ids, axis=0)

        merged_dict = {}
        for key, val in data_dict.items():
            if key in ['pred_boxes', 'pred_scores', 'pred_labels']:
                merged_data = []
                for data in val:
                    data = data.cpu().numpy()
                    merged_data.append(data)
                merged_dict[key] = np.concatenate(merged_data, axis=0)

        pert_det_boxes = merged_dict['pred_boxes']
        pert_det_labels = merged_dict['pred_labels']
        pert_det_scores = merged_dict['pred_scores']
        return pert_det_boxes, pert_det_labels, pert_det_scores, batch_ids

    def compute_iou(self, base_boxes, pert_boxes):
        """
        3D IoU between base and perturbed detections
        """
        base_boxes = torch.from_numpy(base_boxes)
        pert_boxes = torch.from_numpy(pert_boxes)
        base_boxes, pert_boxes = base_boxes.cuda(), pert_boxes.cuda()
        iou = boxes_iou3d_gpu(base_boxes, pert_boxes)
        iou = iou.cpu().numpy()
        return iou

    def compute_translation_score(self, base_boxes, pert_boxes):
        """
        translation score (see paper for details)
        """
        translation_error = np.linalg.norm(
            base_boxes[:, :3][:, None, :] - pert_boxes[:, :3], axis=2)
        translation_score = 1 - translation_error
        translation_score[translation_score < 0] = 0
        return translation_score

    def compute_orientation_score(self, base_boxes, pert_boxes):
        """
        orientation score (see paper for details)
        """
        boxes_a = copy.deepcopy(base_boxes)
        boxes_b = copy.deepcopy(pert_boxes)

        boxes_a[:, 6] = boxes_a[:, 6] % (2 * math.pi)
        boxes_a[boxes_a[:, 6] > math.pi, 6] -= 2 * math.pi
        boxes_a[boxes_a[:, 6] < -math.pi, 6] += 2 * math.pi
        boxes_b[:, 6] = boxes_b[:, 6] % (2 * math.pi)
        boxes_b[boxes_b[:, 6] > math.pi, 6] -= 2 * math.pi
        boxes_b[boxes_b[:, 6] < -math.pi, 6] += 2 * math.pi
        orientation_error_ = np.abs(
            boxes_a[:, 6][:, None] - boxes_b[:, 6][None, :])
        orientation_error__ = 2 * math.pi - np.abs(
            boxes_a[:, 6][:, None] - boxes_b[:, 6][None, :])
        orientation_error = np.concatenate(
            (orientation_error_[:, :, None], orientation_error__[:, :, None]),
            axis=2)
        orientation_error = np.min(orientation_error, axis=2)
        orientation_score = 1 - orientation_error
        orientation_score[orientation_score < 0] = 0
        return orientation_score

    def compute_orientation_score_new(self, base_boxes, pert_boxes):
        """
        Modified orientation score calculation.
        The score is calculated as 1 - normalized angle difference.
        - Angle difference 0 or pi -> score 1
        - Angle difference pi/2 -> score 0
        """
        boxes_a = copy.deepcopy(base_boxes)
        boxes_b = copy.deepcopy(pert_boxes)

        # Normalize angles to be within [-pi, pi]
        boxes_a[:, 6] = boxes_a[:, 6] % (2 * math.pi)
        boxes_a[boxes_a[:, 6] > math.pi, 6] -= 2 * math.pi
        boxes_a[boxes_a[:, 6] < -math.pi, 6] += 2 * math.pi
        boxes_b[:, 6] = boxes_b[:, 6] % (2 * math.pi)
        boxes_b[boxes_b[:, 6] > math.pi, 6] -= 2 * math.pi
        boxes_b[boxes_b[:, 6] < -math.pi, 6] += 2 * math.pi

        # Compute orientation difference
        orientation_diff = np.abs(boxes_a[:, 6][:, None] - boxes_b[:, 6][None, :])
        orientation_diff = np.minimum(orientation_diff, 2 * math.pi - orientation_diff)

        # Normalize orientation difference for score calculation
        orientation_score = np.ones_like(orientation_diff)

        # Score calculation:
        # - For 0 <= orientation_diff <= pi/2: linear transition from 1 to 0
        # - For pi/2 < orientation_diff <= pi: linear transition from 0 to 1
        mask_first_half = orientation_diff <= math.pi / 2
        mask_second_half = orientation_diff > math.pi / 2

        # First half: (0 to pi/2) -> score from 1 to 0
        orientation_score[mask_first_half] = 1 - (orientation_diff[mask_first_half] / (math.pi / 2))

        # Second half: (pi/2 to pi) -> score from 0 to 1
        orientation_score[mask_second_half] = (orientation_diff[mask_second_half] - math.pi / 2) / (math.pi / 2)

        return orientation_score

    def compute_scale_score(self, base_boxes, pert_boxes):
        """
        scale score (see paper for details)
        """
        boxes_centered_a = copy.deepcopy(base_boxes)
        boxes_centered_b = copy.deepcopy(pert_boxes)
        boxes_centered_a[:, :3] = 0
        boxes_centered_a[:, 6] = 0
        boxes_centered_b[:, :3] = 0
        boxes_centered_b[:, 6] = 0
        scale_score = self.compute_iou(boxes_centered_a, boxes_centered_b)
        scale_score[scale_score < 0] = 0
        return scale_score

    def get_similarity_matrix(self, base_det_boxes, base_det_labels,
                              pert_det_boxes, pert_det_labels, pert_det_scores):
        """
        compute similarity score between the base detections in the full
        point cloud and the detections in the perturbed samples

        Parameters
        ----------
        base_det_boxes : (K, 7)
            bounding boxes of detected objects in full pcl
        base_det_labels : (K)
            class labels of detected objects in full pcl
        pert_det_boxes : ndarray (L, 7)
            bounding boxes of all L detections in the perturbed samples of the batch
        pert_det_labels : ndarray (L)
            labels of all L detections in the perturbed samples of the batch
        pert_det_scores : ndarray (L)
            scores of all L detections in the perturbed samples of the batch
        Returns
        -------
        sim_scores : ndarray (K, L)
            similarity score between all K detections in the full pcl and
            the L detections in the perturbed samples within the batch
        """
        if len(pert_det_boxes) == 0:
            return np.zeros((base_det_boxes.shape[0], 0))
        # similarity score is only greater zero if boxes overlap
        s_overlap = self.compute_iou(base_det_boxes, pert_det_boxes) > 0
        s_overlap = s_overlap.astype(np.float32)

        # similarity score is only greater zero for boxes of same class
        s_class = base_det_labels[:, None] == pert_det_labels[None, :]
        s_class = s_class.astype(np.float32)

        # confidence score is directly used (see paper)
        s_conf = np.repeat(pert_det_scores[None, :], base_det_boxes.shape[0], axis=0)

        s_transl = self.compute_translation_score(base_det_boxes, pert_det_boxes)

        s_orient = self.compute_orientation_score(base_det_boxes, pert_det_boxes)
        # s_orient = self.compute_orientation_score_new(base_det_boxes, pert_det_boxes)

        s_score = self.compute_scale_score(base_det_boxes, pert_det_boxes)

        sim_scores = s_overlap * s_conf * s_transl * s_orient * s_score * s_class

        return sim_scores

    def compute_attribution_maps(self, pcl, base_det_boxes, base_det_labels,
                                 batch_size, num_workers):
        """
        attribution map computation for each base detection

        Parameters
        ----------
        pcl : ndarray (N, 4)
            preprocessed full point cloud (x, y, z, intensity)
        base_det_boxes : ndarray (K, 7)
            bounding boxes of detected objects in full pcl
        base_det_labels : ndarray (K)
            class labels of detected objects in full pcl
        batch_size : int
            batch_size during AM computation
        num_workers : int
            number of dataloader workers

        Returns
        -------
        attr_maps : ndarray (K, N)
            attribution scores for all K detected base objects and all N points
        """

        attr_maps = np.zeros((base_det_labels.shape[0], pcl.shape[0]))
        # count number of occurrences of each point in sampled pcl's
        sampling_map = np.zeros(pcl.shape[0])

        occam_inference_dataset = OccamInferenceDataset(
            data_config=self.data_config, class_names=self.class_names,
            occam_config=self.occam_config, pcl=pcl, nr_it=self.nr_it,
            logger=self.logger
        )

        # pytorch的dataloader
        dataloader = DataLoader(
            occam_inference_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, shuffle=False,
            collate_fn=occam_inference_dataset.collate_batch, drop_last=False,
            sampler=None, timeout=0
        )

        progress_bar = tqdm.tqdm(
            total=self.nr_it, leave=True, desc='OccAM computation',
            dynamic_ncols=True)

        with torch.no_grad():
            # 这里的batch_dict已经是mask过的了
            # 这个enumerate内部会调用OccamInferenceDataset.__getitem__()
            for i, batch_dict in enumerate(dataloader):
                # print(i)
                load_data_to_gpu(batch_dict)
                pert_pred_dicts, _ = self.model.forward(batch_dict)
                # 这一步是把一个batch里所有的检测结果放在一起，batch_ids用来描述某个检测结果属于哪个batch
                # 总的数量会变化，如有的batch里可能一共有16个检测结果，有的可能只有14个等等
                pert_det_boxes, pert_det_labels, pert_det_scores, batch_ids = \
                    self.merge_detections_in_batch(pert_pred_dicts)

                similarity_matrix = self.get_similarity_matrix(
                    base_det_boxes, base_det_labels,
                    pert_det_boxes, pert_det_labels, pert_det_scores)

                # 这里长度固定是8 batch_size
                cur_batch_size = len(pert_pred_dicts)
                for j in range(cur_batch_size):
                    cur_mask = batch_dict['mask'][j, :].cpu().numpy()
                    sampling_map += cur_mask

                    batch_sample_mask = batch_ids == j

                    if np.sum(batch_sample_mask) > 0:
                        max_score = np.max(
                            similarity_matrix[:, batch_sample_mask], axis=1)
                        attr_maps += max_score[:, None] * cur_mask

                progress_bar.update(n=cur_batch_size)

        progress_bar.close()

        # normalize using occurrences
        attr_maps[:, sampling_map > 0] /= sampling_map[sampling_map > 0]

        return attr_maps

    def visualize_attr_map(self, points, box, attr_map, draw_origin=False):
        turbo_cmap = plt.get_cmap('turbo')
        attr_map_scaled = attr_map - attr_map.min()
        attr_map_scaled /= attr_map_scaled.max()
        color = turbo_cmap(attr_map_scaled)[:, :3]

        vis = open3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().point_size = 4.0
        vis.get_render_option().background_color = np.ones(3) * 0.25

        if draw_origin:
            axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis_pcd)

        rot_mat = Rotation.from_rotvec([0, 0, box[6]]).as_matrix()
        bb = open3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
        bb.color = (1.0, 0.0, 1.0)
        vis.add_geometry(bb)

        pts = open3d.geometry.PointCloud()
        pts.points = open3d.utility.Vector3dVector(points[:, :3])
        pts.colors = open3d.utility.Vector3dVector(color)
        vis.add_geometry(pts)

        vis.run()
        vis.destroy_window()

    """
        ------------------------------------------------------------------------------
        The above code is Occam's source code, and the following is implemented by Kexuan Xia.
    """

    def generate_random_combination(self, seed=0):
        """
        Generate two random lists of indices.
        """
        arr_1, arr_2 = np.arange(self.nr_it), np.arange(self.nr_it)
        np.random.seed(seed)
        np.random.shuffle(arr_1)
        np.random.shuffle(arr_2)
        result = np.column_stack((arr_1, arr_2))
        return result

    def read_and_preprocess_masked_dt_results(self, source_file_path):
        """
        Read and pre-process masked detection results.
        """
        read_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_masked_dt_results/{source_file_path[-10: -4]}_{self.nr_it}.pkl'
        masked_dt_results = []
        with open(read_path, 'rb') as file:
            data = pickle.load(file)
        for dict in data:
            pred_boxes = dict["box3d_lidar"]
            pred_boxes[:, [3, 4]] = pred_boxes[:, [4, 3]]
            pred_scores = dict["scores"]
            pred_labels = dict["label_preds"] + 1
            if pred_labels.shape == (0, 4):
                pred_labels = pred_labels.reshape(-1)
            for i in range(pred_boxes.shape[0]):
                pred_boxes[i, 6] = -pred_boxes[i, 6] - np.pi / 2
                pred_boxes[i, 2] = pred_boxes[i, 2] + pred_boxes[i, 5] / 2
            masked_dt_results.append({
                "pred_boxes": pred_boxes,
                "pred_scores": pred_scores,
                "pred_labels": pred_labels
            })

        #read_path = f'/home/xkx/kitti/training/velodyne_masked/{source_file_path[-10: -4]}_{self.nr_it}.pkl'
        read_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_masked_pointcloud/{source_file_path[-10: -4]}_{self.nr_it}.pkl'
        mask = []
        # mask是按照batch_size 8个8个一组的
        with open(read_path, 'rb') as file:
            data = pickle.load(file)
        for dict in data:
            mask.append(
                {"mask": dict["mask"]}
            )
        print(f"Successfully read masked detection results from {read_path}")
        return masked_dt_results, mask

    def read_and_preprocess_masked_dt_results_random_combination(self, source_file_path):
        """
        Read and pre-process masked detection results, these detection results are generated by using random combination strategy.
        """
        read_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/Random_combination/'
                     f'CLOC_detection_results/{source_file_path[-10: -4]}_{self.nr_it}.pkl')
        masked_dt_results = []
        with open(read_path, 'rb') as file:
            data = pickle.load(file)
        for dict in data:
            pred_boxes = dict["box3d_lidar"]
            pred_boxes[:, [3, 4]] = pred_boxes[:, [4, 3]]
            pred_scores = dict["scores"]
            pred_labels = dict["label_preds"] + 1
            if pred_labels.shape == (0, 4):
                pred_labels = pred_labels.reshape(-1)
            for i in range(pred_boxes.shape[0]):
                pred_boxes[i, 6] = -pred_boxes[i, 6] - np.pi / 2
                pred_boxes[i, 2] = pred_boxes[i, 2] + pred_boxes[i, 5] / 2
            masked_dt_results.append({
                "pred_boxes": pred_boxes,
                "pred_scores": pred_scores,
                "pred_labels": pred_labels
            })

        read_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_masked_pointcloud/{source_file_path[-10: -4]}_{self.nr_it}.pkl'
        mask = []
        with open(read_path, 'rb') as file:
            data = pickle.load(file)
        for dict in data:
            # batch_size = 8
            for i in range(8):
                mask.append(dict["mask"][i, :])

        print(f"Successfully read masked detection results from {read_path}")
        return masked_dt_results, mask

    # read unmasked detection results from CLOCs
    def read_original_dt_results(self, source_file_path):
        """
        Read detection results without any masks from disk.
        """
        #read_path = f'/home/xkx/kitti/training/velodyne_masked_dt_results/{source_file_path[-10: -4]}_original.pkl'
        read_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_original_dt_results/{source_file_path[-10: -4]}_original.pkl'
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

    def compute_attribution_maps_fusion(self, pcl, base_det_boxes, base_det_labels,
                                        batch_size, source_file_path):

        attr_maps = np.zeros((base_det_labels.shape[0], pcl.shape[0]))
        # count number of occurrences of each point in sampled pcl's
        sampling_map = np.zeros(pcl.shape[0])
        scores = np.zeros((base_det_labels.shape[0], self.nr_it))

        if len(base_det_labels) == 0:
            return attr_maps, scores

        progress_bar = tqdm.tqdm(
            total=self.nr_it, leave=True, desc='OccAM computation',
            dynamic_ncols=True)

        # masked_dt_results: 3000, mask: 375
        masked_dt_results, mask = self.read_and_preprocess_masked_dt_results(source_file_path)

        # 3000/8=375
        for i in range(self.nr_it // batch_size):
            pert_pred_dicts = masked_dt_results[i * batch_size: (i + 1) * batch_size]
            print(pert_pred_dicts[0]['pred_boxes'].cpu().numpy())
            batch_mask = mask[i]
            load_data_to_gpu(batch_mask)

            pert_det_boxes, pert_det_labels, pert_det_scores, batch_ids = \
                self.merge_detections_in_batch(pert_pred_dicts)

            similarity_matrix = self.get_similarity_matrix(
                base_det_boxes, base_det_labels,
                pert_det_boxes, pert_det_labels, pert_det_scores)

            cur_batch_size = len(pert_pred_dicts)
            for j in range(cur_batch_size):
                cur_mask = batch_mask['mask'][j, :].cpu().numpy()
                sampling_map += cur_mask

                batch_sample_mask = batch_ids == j
                if np.sum(batch_sample_mask) > 0:
                    temp = similarity_matrix[:, batch_sample_mask]
                    max_score = np.max(temp, axis=1)
                    attr_maps += max_score[:, None] * cur_mask
                    scores[:, i * batch_size + j] = max_score

            progress_bar.update(n=cur_batch_size)

        progress_bar.close()

        # normalize using occurrences
        attr_maps[:, sampling_map > 0] /= sampling_map[sampling_map > 0]

        return attr_maps, scores

    def compute_attribution_maps_fusion_random_combination(self, pcl, base_det_boxes, base_det_labels,
                                                           batch_size, source_file_path):
        attr_maps = np.zeros((base_det_labels.shape[0], pcl.shape[0]))
        # count number of occurrences of each point in sampled pcl's
        sampling_map = np.zeros(pcl.shape[0])

        if len(base_det_labels) == 0:
            return attr_maps

        progress_bar = tqdm.tqdm(
            total=self.nr_it, leave=True, desc='OccAM computation',
            dynamic_ncols=True)

        masked_dt_results, masks = self.read_and_preprocess_masked_dt_results_random_combination(source_file_path)
        mask_index = self.generate_random_combination(seed=0)[:, 0]

        for i in range(self.nr_it):
            pert_pred_dicts = masked_dt_results[i]

            pert_det_boxes = pert_pred_dicts['pred_boxes'].cpu().numpy()
            pert_det_labels = pert_pred_dicts['pred_labels'].cpu().numpy()
            pert_det_scores = pert_pred_dicts['pred_scores'].cpu().numpy()
            # print(f'pert_pred_boxes: {pert_det_boxes}')
            # print(f'pert_det_scores: {pert_det_scores}')

            similarity_matrix = self.get_similarity_matrix(
                base_det_boxes, base_det_labels,
                pert_det_boxes, pert_det_labels, pert_det_scores)

            mask = masks[mask_index[i]]
            sampling_map += mask

            if similarity_matrix.shape[1] != 0:
                max_score = np.max(similarity_matrix, axis=1)
                attr_maps += max_score[:, None] * mask

            progress_bar.update(n=i)

        progress_bar.close()

        # normalize using occurrences
        attr_maps[:, sampling_map > 0] /= sampling_map[sampling_map > 0]

        return attr_maps

    def save_masked_input(self, save_path, pcl, batch_size, num_workers):

        occam_inference_dataset = OccamInferenceDataset(
            data_config=self.data_config, class_names=self.class_names,
            occam_config=self.occam_config, pcl=pcl, nr_it=self.nr_it,
            logger=self.logger
        )

        # pytorch的dataloader
        dataloader = DataLoader(
            occam_inference_dataset, batch_size=batch_size, pin_memory=True,
            num_workers=num_workers, shuffle=False,
            collate_fn=occam_inference_dataset.collate_batch, drop_last=False,
            sampler=None, timeout=0
        )

        results = []

        for i, dict in enumerate(dataloader):
            dict_to_save = {
                'points': dict['points'],
                'mask': dict['mask'],
                'vx_orig_coord': dict['vx_orig_coord'],
                'vx_keep_ids': dict['vx_keep_ids']
            }
            results.append(dict_to_save)

        with open(save_path, "wb") as file:
            pickle.dump(results, file)

        # print("results.shape: ", len(results))
        # print("points.shape: ", results[0]["points"].shape)
        # print("mask.shape: ", results[0]["mask"].shape)
        # print("results.example: ", results[0])

        print(f"Masked {save_path[-15:]} has been stored")

    def save_pt_vx_id(self, save_path, pcl):
        """
        Save the correspondence between points and voxels.
        """

        occam_inference_dataset = OccamInferenceDataset(
            data_config=self.data_config, class_names=self.class_names,
            occam_config=self.occam_config, pcl=pcl, nr_it=self.nr_it,
            logger=self.logger
        )

        pt_vx_id = occam_inference_dataset.get_pt_vx_id()
        print(pt_vx_id.shape)
        print(pt_vx_id)

        np.save(save_path, pt_vx_id)

        print(f"pt_vx_id for {save_path[-15:]} has been stored")

    """
    Trying to integrate CLOCs into OccAM
    """

    def build_example(self, source_file_path, anchor_cache=None):
        """
        Build the parameter "example" for method "build_tensor_input".
        """
        idx = int(source_file_path[-10: -4])
        info = read_kitti_info_val(idx)
        rect = info['calib/R0_rect']
        P2 = info['calib/P2']
        Trv2c = info['calib/Tr_velo_to_cam']

        example = {
            'rect': rect,
            'Trv2c': Trv2c,
            'P2': P2,
            'image_shape': np.array(info["img_shape"], dtype=np.int32),
            'image_idx': info['image_idx'],
            'image_path': info['img_path'],
        }

        # The parameter values is only valid for SECOND detector
        feature_map_size = [1, 200, 176]
        # feature_map_size = [*feature_map_size, 1][::-1]
        if anchor_cache is not None:
            anchors = anchor_cache["anchors"]
        else:
            # a = self.anchor_ranges
            ret = generate_anchors(feature_map_size, self.anchor_ranges)
            anchors = ret["anchors"]
            anchors = anchors.reshape([-1, 7])
        example["anchors"] = anchors
        example = merge_second_batch([example])
        example = example_convert_to_torch(example)
        return example

    def build_tensor_input(self, example, preds_dict, top_predictions):
        batch_size = example['anchors'].shape[0]
        batch_anchors = example["anchors"].view(batch_size, -1, 7)
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        batch_image_shape = example["image_shape"]
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
        batch_imgidx = example['image_idx']

        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1, self.code_size)
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1
        batch_cls_preds = batch_cls_preds.view(batch_size, -1, num_class_with_bg)
        batch_box_preds = decode_torch(batch_box_preds, batch_anchors)

        # if self._use_direction_classifier:
        #     batch_dir_preds = preds_dict["dir_cls_preds"]
        #     batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
        # else:
        #     batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        for box_preds, cls_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_rect,
                batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            rect = rect.float()
            Trv2c = Trv2c.float()
            P2 = P2.float()

            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
                #total_scores = cls_preds   # use this if you want to fuse raw log score
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]

            # finally generate predictions.
            final_box_preds = box_preds
            final_scores = total_scores
            final_box_preds_camera = utils.box_lidar_to_camera(final_box_preds, rect, Trv2c)
            locs = final_box_preds_camera[:, :3]
            dims = final_box_preds_camera[:, 3:6]
            angles = final_box_preds_camera[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            box_corners = utils.center_to_corner_box3d(
                locs, dims, angles, camera_box_origin, axis=1)

            box_corners_in_image = utils.project_to_image(
                box_corners, P2)
            # box_corners_in_image: [N, 8, 2]
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            img_height = batch_image_shape[0, 0]
            img_width = batch_image_shape[0, 1]
            minxy[:, 0] = torch.clamp(minxy[:, 0], min=0, max=img_width)
            minxy[:, 1] = torch.clamp(minxy[:, 1], min=0, max=img_height)
            maxxy[:, 0] = torch.clamp(maxxy[:, 0], min=0, max=img_width)
            maxxy[:, 1] = torch.clamp(maxxy[:, 1], min=0, max=img_height)
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            # predictions
            predictions_dict = {
                "bbox": box_2d_preds,
                "box3d_camera": final_box_preds_camera,
                "box3d_lidar": final_box_preds,
                "scores": final_scores,
                #"label_preds": label_preds,
                "image_idx": img_idx,
            }
            predictions_dicts.append(predictions_dict)
            dis_to_lidar = torch.norm(box_preds[:, :2], p=2, dim=1, keepdim=True) / 82.0
            box_2d_detector = np.zeros((200, 4))
            box_2d_detector[0:top_predictions.shape[0], :] = top_predictions[:, :4]
            box_2d_detector = top_predictions[:, :4]
            box_2d_scores = top_predictions[:, 4].reshape(-1, 1)
            overlaps1 = np.zeros((900000, 4), dtype=box_2d_preds.detach().cpu().numpy().dtype)
            tensor_index1 = np.zeros((900000, 2), dtype=box_2d_preds.detach().cpu().numpy().dtype)
            overlaps1[:, :] = -1
            tensor_index1[:, :] = -1
            #final_scores[final_scores<0.1] = 0
            #box_2d_preds[(final_scores<0.1).reshape(-1),:] = 0
            iou_test, tensor_index, max_num = build_stage2_training(box_2d_preds.detach().cpu().numpy(),
                                                                    box_2d_detector,
                                                                    -1,
                                                                    final_scores.detach().cpu().numpy(),
                                                                    box_2d_scores,
                                                                    dis_to_lidar.detach().cpu().numpy(),
                                                                    overlaps1,
                                                                    tensor_index1)
            iou_test_tensor = torch.FloatTensor(iou_test)  #iou_test_tensor shape: [160000,4]
            tensor_index_tensor = torch.LongTensor(tensor_index)
            iou_test_tensor = iou_test_tensor.permute(1, 0)
            iou_test_tensor = iou_test_tensor.reshape(1, 4, 1, 900000)
            tensor_index_tensor = tensor_index_tensor.reshape(-1, 2)
            if max_num == 0:
                non_empty_iou_test_tensor = torch.zeros(1, 4, 1, 2)
                non_empty_iou_test_tensor[:, :, :, :] = -1
                non_empty_tensor_index_tensor = torch.zeros(2, 2)
                non_empty_tensor_index_tensor[:, :] = -1
            else:
                non_empty_iou_test_tensor = iou_test_tensor[:, :, :, :max_num]
                non_empty_tensor_index_tensor = tensor_index_tensor[:max_num, :]

        return predictions_dicts, non_empty_iou_test_tensor, non_empty_tensor_index_tensor

    def get_fused_base_detection(self, source_file_path):
        """
        Feed raw image and lidar points into SECOND and 2d detector(here is CasCade RCNN) respectively.
        Take 3D proposals and 2D detection results as input for fusion layer.
        Fuse them following the method and code from CLOCs.
        """
        input_dict = {
            'points': self.load_and_preprocess_pcl(source_file_path)
        }

        data_dict = self.base_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.base_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        # SECOND forward function
        with torch.no_grad():
            pred_dict = self.model.forward_for_fusion(data_dict)

        proposal_3d = {
            "box_preds": pred_dict['batch_box_preds'].reshape(1, 200, 176, 14),
            "cls_preds": pred_dict['batch_cls_preds'].reshape(1, 200, 176, 2)
        }

        # read_path = f'/home/xkx/桌面/example.pkl'
        # with open(read_path, 'rb') as output_file:
        #     example_1 = pickle.load(output_file)
        # read_path = f'/home/xkx/桌面/preds_dict.pkl'
        # with open(read_path, 'rb') as output_file:
        #     proposal_3d_1 = pickle.load(output_file)
        # read_path = f'/home/xkx/桌面/top_predictions.pkl'
        # with open(read_path, 'rb') as output_file:
        #     proposal_2d_1 = pickle.load(output_file)

        # get 2d proposals(actually it's final detection results)
        proposal_2d = self.get_proposal_2d(source_file_path)
        # build "example", which contains basic information for the scene
        example = self.build_example(source_file_path)
        # get fusion input and torch index, which are necessary for fusion layer
        _, fusion_input, torch_index = self.build_tensor_input(example, proposal_3d, proposal_2d)

        fusion_model_dir = 'CLOCs/CLOCs_SecCas_pretrained'
        fusion_layer = fusion()
        fusion_layer.cuda()
        torchplus.train.try_restore_latest_checkpoints(fusion_model_dir, [fusion_layer])
        fusion_layer.eval()

        # call fusion layer and forward
        fusion_cls_preds, flag = fusion_layer(fusion_input.cuda(), torch_index.cuda())
        fusion_cls_preds_reshape = fusion_cls_preds.reshape(1, 200, 176, 2)
        data_dict['batch_cls_preds'] = fusion_cls_preds
        base_det = self.model.post_processing(data_dict)
        # TODO: Now, the base_det doesn't show good results
        # TODO: According to what I tested, the method "build_tensor_input", the parameters "example", "proposal_2d" work well.
        # TODO: The problem seems to happen in the proposal_3d

        return base_det

    def get_proposal_2d(self, source_file_path):
        """
        Get 2D detection proposals by calling 2D detection APIs from mmdetection
        As the author of CLOCs claimed, using detection results instead of proposals only make small impact on the fusion results.
        From now on, I cannot get 2D proposals so I use 2D results as input for the 2D pipeline.
        """
        inferencer = DetInferencer(model='mmdetection/configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py',
                                   weights='mmdetection/checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth')

        idx = int(source_file_path[-10: -4])
        image_path = f'/home/xkx/kitti/training/image_2/{str(idx).zfill(6)}.png'

        # Perform inference
        prediction_result = inferencer(image_path)['predictions'][0]

        labels = np.array(prediction_result['labels'])
        scores = np.array(prediction_result['scores'])
        bboxes = np.array(prediction_result['bboxes'])

        # Now I can only find pre-trained model on COCO dataset, the class label in COCO dataset is different from KITTI
        if self.target_class == 'car':
            target_label = 2
        else:
            return "Need to check the pedestrian and cyclist class label in COCO dataset"

        indices = np.where(labels == target_label)
        labels = labels[indices]
        scores = np.round(scores[indices], 4)
        bboxes = np.round(bboxes[indices], 2)

        proposal_2d = np.column_stack((bboxes, scores))

        return proposal_2d

    #TODO: Implement a new method by imitating the method "compute_attribution_maps" but adapt it according to fusion
