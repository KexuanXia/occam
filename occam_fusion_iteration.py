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
    #                     default='/home/xkx/kitti/training/velodyne/000000.bin',
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

    # 读取model_cfg_file和occam_cfg_file的内容并把它们拼接成一个字典
    # 因为model_cfg_file里包含了_BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
    # 所以实际上是kitti_dataset.yaml, second.yaml, kitti_pointpillar.yaml三个配置文件的拼接
    cfg_from_yaml_file(args.model_cfg_file, cfg)
    cfg_from_yaml_file(args.occam_cfg_file, cfg)

    return args, cfg


def main(start_idx, end_idx):
    args, config = parse_config()
    logger = common_utils.create_logger()
    logger.info('------------------------ OccAM_Fusion Demo -------------------------')

    non_car_scene = []
    for idx in range(start_idx, end_idx):
        source_file_path = '/home/xkx/kitti/training/velodyne/'
        idx_str = str(idx).zfill(6)
        source_file_path = source_file_path + idx_str + '.bin'

        occam = OccAM(data_config=config.DATA_CONFIG, model_config=config.MODEL,
                      occam_config=config.OCCAM, class_names=config.CLASS_NAMES,
                      model_ckpt_path=args.ckpt, nr_it=args.nr_it, logger=logger)

        pcl = occam.load_and_preprocess_pcl(source_file_path)

        base_det_boxes, base_det_labels, base_det_scores = occam.read_original_dt_results(source_file_path)

        attr_maps = occam.compute_attribution_maps_fusion(
            pcl=pcl, base_det_boxes=base_det_boxes,
            base_det_labels=base_det_labels, batch_size=args.batch_size, source_file_path=source_file_path)

        print(f"pcl.shape: {pcl.shape}")
        print(f"attr_maps.shape: {attr_maps.shape}")
        print(f"attr_maps: {attr_maps}")
        print(f"max in attr_maps: {np.amax(attr_maps, axis=1)}")

        save_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/heat_map/{source_file_path[-10: -4]}_{args.nr_it}.pkl'
        with open(save_path, "wb") as file:
            pickle.dump(attr_maps, file)
    print(f"non_car_scene: {non_car_scene}")
    logger.info('finished')


if __name__ == '__main__':
    main(0, 13)
