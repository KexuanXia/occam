"""
    This file generate 3D heat maps for random combination strategy.
"""


import argparse

import numpy as np
import pickle

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

from occam_utils.occam import OccAM


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_cfg_file', type=str,
                        default='cfgs/kitti_models/second.yaml',
                        help='dataset/model config for the demo')
    parser.add_argument('--occam_cfg_file', type=str,
                        default='cfgs/occam_configs/kitti_pointpillar.yaml',
                        help='specify the OccAM config')
    # parser.add_argument('--source_file_path', type=str,
    #                     default='/home/xkx/kitti/training/velodyne/000007.bin',
    #                     help='point cloud data file to analyze')
    parser.add_argument('--ckpt', type=str,
                        default='pretrained_model/based_on_kitti/second_7862.pth', required=False,
                        help='path to pretrained model parameters')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for OccAM creation')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for dataloader')
    parser.add_argument('--nr_it', type=int, default=3000,
                        help='number of sub-sampling iterations N')
    parser.add_argument('--object', type=int, default=0,
                        help='number of detected object')

    args = parser.parse_args()

    cfg_from_yaml_file(args.model_cfg_file, cfg)
    cfg_from_yaml_file(args.occam_cfg_file, cfg)

    return args, cfg


def main(start_idx, end_idx, save=False, show=False):
    args, config = parse_config()
    logger = common_utils.create_logger()
    logger.info('------------------------ OccAM_Fusion Demo -------------------------')

    # iteratively generate heat maps according to index
    for idx in range(start_idx, end_idx):
        source_file_path = '/home/xkx/kitti/training/velodyne/'
        idx_str = str(idx).zfill(6)
        source_file_path = source_file_path + idx_str + '.bin'

        occam = OccAM(data_config=config.DATA_CONFIG, model_config=config.MODEL,
                      occam_config=config.OCCAM, class_names=config.CLASS_NAMES,
                      model_ckpt_path=args.ckpt, nr_it=args.nr_it, logger=logger)

        pcl = occam.load_and_preprocess_pcl(source_file_path)

        base_det_boxes, base_det_labels, base_det_scores = occam.read_original_dt_results(source_file_path)
        print("CLOC base detection")
        print("base_det_boxes: ", base_det_boxes)
        print("base_det_labels: ", base_det_labels)
        print("base_det_scores: ", base_det_scores)

        # generate 3D heat map for random combination
        attr_maps_random_combination = occam.compute_attribution_maps_fusion_random_combination(
            pcl=pcl, base_det_boxes=base_det_boxes,
            base_det_labels=base_det_labels, batch_size=args.batch_size, source_file_path=source_file_path)
        print(f"attr_maps.shape: {attr_maps_random_combination.shape}")
        print(f"attr_maps: {attr_maps_random_combination}")
        print(f"max in attr_maps: {np.amax(attr_maps_random_combination, axis=1)}")
        print(f"min in attr_maps: {np.amin(attr_maps_random_combination, axis=1)}")

        if save:
            save_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/Random_combination/occam_heat_map_data/'
                         f'{source_file_path[-10: -4]}_{args.nr_it}.pkl')
            with open(save_path, "wb") as file:
                pickle.dump(attr_maps_random_combination, file)

        logger.info(f'Visualize attribution map of {args.object}th object')
        if show:
            occam.visualize_attr_map(pcl, base_det_boxes[args.object], attr_maps_random_combination[args.object])


if __name__ == '__main__':
    main(15, 17, save=False, show=True)