"""
    This file pre-processes the point clouds from KITTI training dataset according to configs
    and save it on the disk.
    This file also generate 3000 point cloud masks for each scene.
    The saved point clouds and masked point clouds serve as the input for CLOCs detector.
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

        # save cropped point cloud
        save_path = '/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_croped_by_occam/'
        save_path = save_path + idx_str + '.bin'
        pcl.tofile(save_path)

        # save 3000 times masked point cloud
        save_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_masked_pointcloud_2/{source_file_path[-10: -4]}_{args.nr_it}.pkl'
        occam.save_masked_input(save_path, pcl, args.batch_size, args.workers)

    logger.info('finished')


if __name__ == '__main__':
    main(83, 300)

