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
    parser.add_argument('--source_file_path', type=str,
                        default='/home/xkx/kitti/training/velodyne/000011.bin',
                        help='point cloud data file to analyze')
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


def main():
    args, config = parse_config()
    logger = common_utils.create_logger()
    logger.info('------------------------ OccAM_Fusion Demo -------------------------')

    occam = OccAM(data_config=config.DATA_CONFIG, model_config=config.MODEL,
                  occam_config=config.OCCAM, class_names=config.CLASS_NAMES,
                  model_ckpt_path=args.ckpt, nr_it=args.nr_it, logger=logger)

    pcl = occam.load_and_preprocess_pcl(args.source_file_path)

    save_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_masked_pointcloud/{args.source_file_path[-10: -4]}_{args.nr_it}.pkl'
    print(save_path)
    occam.save_masked_input(save_path, pcl, args.batch_size, args.workers)


if __name__ == '__main__':
    main()
