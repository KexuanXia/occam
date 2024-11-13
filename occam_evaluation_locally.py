"""
    Post-process and plot evaluation curves for applying OccAM to CLOCs
"""

import numpy as np
import pickle
import torch
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
import matplotlib.pyplot as plt
import open3d
from scipy.spatial.transform import Rotation


# read evaluation detection results from disk
def read_evaluation_dt_results(idx):
    read_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_evaluation_dt_results'
                 f'/{str(idx).zfill(6)}.pkl')
    with open(read_path, 'rb') as file:
        data = pickle.load(file)

    sort_types = {'descend': 0,
                  'random': 1,
                  'ascend': 2}
    eval_dict = {
        'descend': [],
        'random': [],
        'ascend': []
    }

    if len(data) == 0:
        return eval_dict

    # prediction results in CLOCs are different from the ones in OccAM
    # the form of detection results have to be modified
    for sort_type, value in sort_types.items():
        for i in range(0, 11):
            pred_boxes = data[value][i]["box3d_lidar"]
            pred_boxes[:, [3, 4]] = pred_boxes[:, [4, 3]]
            pred_scores = data[value][i]["scores"]

            pred_boxes = pred_boxes.cpu().numpy()
            pred_scores = pred_scores.cpu().numpy()

            for j in range(pred_boxes.shape[0]):
                pred_boxes[j, 6] = -pred_boxes[j, 6] - np.pi / 2
                pred_boxes[j, 2] = pred_boxes[j, 2] + pred_boxes[j, 5] / 2

            eval_dict[sort_type].append(np.column_stack((pred_boxes, pred_scores)))

    return eval_dict


def post_process_scores_and_boxes(eval_dict, iou_threshold=0.5):
    sort_types = ['descend', 'random', 'ascend']
    score_dict = {
        'descend': [1],
        'random': [1],
        'ascend': [1]
    }
    IoU_dict = {
        'descend': [1],
        'random': [1],
        'ascend': [1]
    }

    # 3 order: descend, random, ascend
    for sort_type in sort_types:
        original_dt_boxes = eval_dict[sort_type][0][:, :7]
        original_dt_scores = eval_dict[sort_type][0][:, 7]
        num_objects = original_dt_boxes.shape[0]
        score, IoU = np.zeros(11), np.zeros(11)
        score[0], IoU[0] = 1, 1

        # from 10% removal to 100% removal, interval is 10%
        for i in range(1, 11):
            eval_dt_boxes = eval_dict[sort_type][i][:, :7]
            eval_dt_scores = eval_dict[sort_type][i][:, 7]
            temp_IoU = np.zeros(num_objects)
            temp_scores = np.zeros(num_objects)
            for object_index, original_dt_box in enumerate(original_dt_boxes):
                original_dt_box = original_dt_box.reshape(1, 7)
                # index_max_IoU = 0
                for eval_object_index, eval_dt_box in enumerate(eval_dt_boxes):
                    eval_dt_box = eval_dt_box.reshape(1, 7)
                    temppp_IoU = compute_iou(original_dt_box, eval_dt_box)[0]
                    if temppp_IoU > temp_IoU[object_index]:
                        temp_IoU[object_index] = temppp_IoU
                        index_max_IoU = eval_object_index
                if temp_IoU[object_index] > iou_threshold:
                    temp_scores[object_index] = eval_dt_scores[index_max_IoU] / original_dt_scores[object_index]
                else:
                    temp_scores[object_index] = 0

                temp_scores[object_index] = 1 if temp_scores[object_index] > 1 else temp_scores[object_index]
            score_dict[sort_type].append(np.mean(temp_scores))
            IoU_dict[sort_type].append(np.mean(temp_IoU))

    return score_dict, IoU_dict


def plot_occam_evaluation_new(score_dict, IoU_dict):
    x = np.arange(0.0, 1.1, 0.1)

    plt.figure(figsize=(10, 8))
    plt.suptitle('Heat Map Evaluation')
    plt.subplot(1, 2, 1)
    plt.plot(x, score_dict['descend'], color='green', label='Descend')
    plt.plot(x, score_dict['random'], color='red', label='Random')
    plt.plot(x, score_dict['ascend'], color='blue', label='Ascend')
    plt.scatter(x, score_dict['descend'], color='green')
    plt.scatter(x, score_dict['random'], color='red')
    plt.scatter(x, score_dict['ascend'], color='blue')
    plt.xlabel('Removed Points')
    plt.ylabel('Mean confidence score')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, IoU_dict['descend'], color='green', label='Descend')
    plt.plot(x, IoU_dict['random'], color='red', label='Random')
    plt.plot(x, IoU_dict['ascend'], color='blue', label='Ascend')
    plt.scatter(x, IoU_dict['descend'], color='green')
    plt.scatter(x, IoU_dict['random'], color='red')
    plt.scatter(x, IoU_dict['ascend'], color='blue')

    plt.xlabel('Removed Points')
    plt.ylabel('Mean IoU')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.show()


# compare the thresholds of 0.5 and 0.7
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
    plt.xlabel('Removed Points')
    plt.ylabel('Mean confidence score')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


# compute iou of two 3D bounding boxes
def compute_iou(boxes_a, boxes_b):
    boxes_a = torch.from_numpy(boxes_a)
    boxes_b = torch.from_numpy(boxes_b)
    boxes_a, boxes_b = boxes_a.cuda(), boxes_b.cuda()
    iou = boxes_iou3d_gpu(boxes_a, boxes_b)
    iou = iou.cpu().numpy()
    return iou


# visualization of heat maps, just for better debug
def visualize_attr_map(points, box, draw_origin=True):
    turbo_cmap = plt.get_cmap('turbo')

    color = [0, 1, 0]
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


# main function: read data from disk, post process, and then plot the curves
def main(start_idx, end_idx, iou_threshold):
    sort_types = ['descend', 'random', 'ascend']
    plot_score_dict = {
        'descend': [],
        'random': [],
        'ascend': []
    }
    plot_IoU_dict = {
        'descend': [],
        'random': [],
        'ascend': []
    }
    for idx in range(start_idx, end_idx):
        eval_dict = read_evaluation_dt_results(idx)
        if len(eval_dict['descend']) == 0: continue
        score_dict, IoU_dict = post_process_scores_and_boxes(eval_dict, iou_threshold)

        for sort_type in sort_types:
            plot_score_dict[sort_type].append(score_dict[sort_type])
            plot_IoU_dict[sort_type].append(IoU_dict[sort_type])

    for sort_type in sort_types:
        plot_score_dict[sort_type] = np.mean(np.array(plot_score_dict[sort_type]), axis=0)
        plot_IoU_dict[sort_type] = np.mean(np.array(plot_IoU_dict[sort_type]), axis=0)
    plot_occam_evaluation_new(plot_score_dict, plot_IoU_dict)


# main function of plotting comparison curves
def compare_two_threshold(start_idx, end_idx):
    sort_types = ['descend', 'random', 'ascend']
    plot_score_dict_05 = {
        'descend': [],
        'random': [],
        'ascend': []
    }
    plot_score_dict_07 = {
        'descend': [],
        'random': [],
        'ascend': []
    }

    for idx in range(start_idx, end_idx):
        eval_dict = read_evaluation_dt_results(idx)
        if len(eval_dict['descend']) == 0: continue
        score_dict, _ = post_process_scores_and_boxes(eval_dict, iou_threshold=0.5)

        for sort_type in sort_types:
            plot_score_dict_05[sort_type].append(score_dict[sort_type])

    for idx in range(start_idx, end_idx):
        eval_dict = read_evaluation_dt_results(idx)
        if len(eval_dict['descend']) == 0: continue
        score_dict, _ = post_process_scores_and_boxes(eval_dict, iou_threshold=0.7)

        for sort_type in sort_types:
            plot_score_dict_07[sort_type].append(score_dict[sort_type])

    for sort_type in sort_types:
        plot_score_dict_05[sort_type] = np.mean(np.array(plot_score_dict_05[sort_type]), axis=0)
        plot_score_dict_07[sort_type] = np.mean(np.array(plot_score_dict_07[sort_type]), axis=0)
    plot_threshold_comparison(plot_score_dict_05, plot_score_dict_07)


if __name__ == '__main__':
    # main(0, 2001, iou_threshold=0.7)
    compare_two_threshold(0, 2000)