"""
    This file save the voxel information when sub-sampling point clouds.
    It should be merged and done in the save_masked_pointcloud.py, but at the beginning of the work,
    I didn't realize that the voxel information will be useful in the heat map combination stage.
    During the mask generation, the 3D space is firstly divided into thousands of voxels. Each
    point belongs to a voxel, and each voxel may have 0 or 1 or multiple points.
    pt_vx_id contains the information that each point belongs to which voxel.
"""


import argparse
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
    #                     default='/home/xkx/kitti/training/velodyne/',
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


def main(start_idx, end_idx):
    args, config = parse_config()
    logger = common_utils.create_logger()
    logger.info('------------------------ OccAM_Fusion Demo -------------------------')

    for idx in range(start_idx, end_idx):
        source_file_path = '/home/xkx/kitti/training/velodyne/'
        idx_str = str(idx).zfill(6)
        source_file_path = source_file_path + idx_str + '.bin'

        occam = OccAM(data_config=config.DATA_CONFIG, model_config=config.MODEL,
                      occam_config=config.OCCAM, class_names=config.CLASS_NAMES,
                      model_ckpt_path=args.ckpt, nr_it=args.nr_it, logger=logger)

        pcl = occam.load_and_preprocess_pcl(source_file_path)

        # save voxel information when generating masks
        save_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/pt_vx_id/{source_file_path[-10: -4]}'
        occam.save_pt_vx_id(save_path, pcl)

    logger.info('finished')


if __name__ == '__main__':
    main(0, 100)

