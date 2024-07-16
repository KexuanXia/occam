import argparse
import pickle

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils

from occam_utils.occam import OccAM

from CLOCs.second.pytorch.models import fusion
from CLOCs import torchplus
import numpy as np


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_cfg_file', type=str,
                        default='cfgs/kitti_models/second.yaml',
                        help='dataset/model config for the demo')
    parser.add_argument('--occam_cfg_file', type=str,
                        default='cfgs/occam_configs/kitti_pointpillar.yaml',
                        help='specify the OccAM config')
    parser.add_argument('--source_file_path', type=str,
                        default='../kitti/training/velodyne/000002.bin',
                        help='point cloud data file to analyze')
    parser.add_argument('--ckpt', type=str,
                        default='pretrained_model/based_on_kitti/second_7862.pth',
                        required=True,
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


def main():
    # args, config = parse_config()
    # logger = common_utils.create_logger()
    # logger.info('------------------------ OccAM Demo -------------------------')

    scene_idx = '000002'
    path_to_pretrained_fusion_model = 'CLOCs/CLOCs_SecCas_pretrained'
    path_to_2d_detection_result = 'CLOCs/d2_detection_data/' + scene_idx + '.txt'
    path_to_example = '../kitti/example/' + scene_idx + '.pkl'
    with open(path_to_example, 'rb') as file:
        example = pickle.load(file)

    # with open(path_to_2d_detection_result, 'r') as f:
    #     lines = f.readlines()
    # # 把检测结果转换成list，每个检测结果为一行
    # content = [line.strip().split(' ') for line in lines]
    # # 检测到的目标类别，这里全是‘Car’
    # predicted_class = np.array([x[0] for x in content], dtype='object')
    # # Car对应的检测索引
    # predicted_class_index = np.where(predicted_class == 'Car')
    # # 提取2d检测框[x1, y1, x2, y2]
    # detection_result = np.array([[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # # 提取检测分数
    # score = np.array([float(x[15]) for x in content])  # 1000 is the score scale!!!
    # # 把检测框和分数放一起[x1, y1, x2, y2, score]
    # # 目前来看f_detection_result, middle_predictions, top_predictions是完全一样的
    # f_detection_result = np.append(detection_result, score.reshape(-1, 1), 1)
    # middle_predictions = f_detection_result[predicted_class_index, :].reshape(-1, 5)
    # top_predictions = middle_predictions[np.where(middle_predictions[:, 4] >= -100)]
    #
    #
    #
    # fusion_layer = fusion.fusion()
    # torchplus.train.try_restore_latest_checkpoints(path_to_pretrained_fusion_model, [fusion_layer])
    # fusion_layer.forward(fusion_input.cuda(),torch_index.cuda())


if __name__ == '__main__':
    main()