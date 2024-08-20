import numpy as np
import pickle
import torch
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import open3d
from scipy.spatial.transform import Rotation
from scipy.optimize import curve_fit


def read_dropped_dt_results(idx):
    read_path = f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/velodyne_dropped_dt_results_three_order/{str(idx).zfill(6)}.pkl'
    with open(read_path, 'rb') as file:
        data = pickle.load(file)

    if len(data) == 0:
        return [], []

    num_objects = data[0][0]["box3d_lidar"].shape[0]
    dropped_dt_boxes = [np.zeros((11, num_objects, 7)) for _ in range(3)]
    dropped_dt_scores = [np.zeros((11, num_objects)) for _ in range(3)]
    # 3 order: descent, random, ascent
    for order in range(3):
        for percentage in range(0, 110, 10):
            i = percentage // 10
            pred_boxes = data[order][i]["box3d_lidar"]
            pred_boxes[:, [3, 4]] = pred_boxes[:, [4, 3]]
            pred_scores = data[order][i]["scores"]

            pred_boxes = pred_boxes.cpu().numpy()
            pred_scores = pred_scores.cpu().numpy()

            # prediction results in CLOCs are different from the ones in OccAM
            for j in range(pred_boxes.shape[0]):
                pred_boxes[j, 6] = -pred_boxes[j, 6] - np.pi / 2
                pred_boxes[j, 2] = pred_boxes[j, 2] + pred_boxes[j, 5] / 2

            num_detected_objects = len(pred_scores)
            if num_detected_objects < num_objects:
                num_padding = num_objects - len(pred_scores)
                array_to_pad_to_scores = np.zeros(num_padding)
                pred_scores = np.concatenate((pred_scores, array_to_pad_to_scores), axis=None)
                array_to_pad_to_boxes = np.zeros((num_padding, 7))
                pred_boxes = np.concatenate((pred_boxes, array_to_pad_to_boxes), axis=0)
            if num_detected_objects > num_objects:
                pred_scores = pred_scores[:num_objects]
                pred_boxes = pred_boxes[:num_objects, :]

            dropped_dt_scores[order][i] = pred_scores
            dropped_dt_boxes[order][i] = pred_boxes
    return dropped_dt_scores, dropped_dt_boxes


def plot_occam_evaluation(line_scores, line_iou):
    x = np.arange(0.0, 1.1, 0.1)

    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.plot(x, line_scores[0], color='blue')
    plt.plot(x, line_scores[1], color='orange')
    plt.plot(x, line_scores[2], color='green')
    plt.scatter(x, line_scores[0])
    plt.scatter(x, line_scores[1])
    plt.scatter(x, line_scores[2])
    plt.xlabel('Removed Points')
    plt.ylabel('Mean confidence score')
    plt.xticks(np.arange(0, 1.1, 0.1))  # 设置 x 轴刻度
    plt.yticks(np.arange(0, 1.1, 0.1))  # 设置 y 轴刻度
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, line_iou[0], color='blue')
    plt.plot(x, line_iou[1], color='orange')
    plt.plot(x, line_iou[2], color='green')
    plt.scatter(x, line_iou[0])
    plt.scatter(x, line_iou[1])
    plt.scatter(x, line_iou[2])

    plt.xlabel('Removed Points')
    plt.ylabel('Mean IOU')
    plt.xticks(np.arange(0, 1.1, 0.1))  # 设置 x 轴刻度
    plt.yticks(np.arange(0, 1.1, 0.1))  # 设置 y 轴刻度
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()

    # 显示图形
    plt.show()


def plot_occam_evaluation_v2(line_scores, line_iou):
    def exponential_func(x, b):
        return np.exp(b * x)

    def exponential_func_2(x, b):
        return - np.exp(b * x) + 2

    x_data = np.arange(0, 1.1, 0.1)
    color = ["blue", "orange", "green"]

    # 绘制原始数据和拟合曲线
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)

    for order in range(3):
        popt, pcov = curve_fit(exponential_func_2, x_data, line_scores[order]) if order==2 else curve_fit(exponential_func, x_data, line_scores[order])

        b = popt[0]

        # 生成拟合曲线
        x_fit = np.linspace(0, 1, 100)
        y_fit = exponential_func_2(x_fit, b) if order == 2 else exponential_func(x_fit, b)

        plt.plot(x_fit, y_fit, color=color[order])
        plt.scatter(x_data, line_scores[order], label='Data')
        plt.legend()

    plt.subplot(1, 2, 2)
    for order in range(3):
        popt, pcov = curve_fit(exponential_func_2, x_data, line_iou[order]) if order == 2 else curve_fit(exponential_func, x_data, line_iou[order])

        b = popt[0]

        # 生成拟合曲线
        x_fit = np.linspace(0, 1, 100)
        y_fit = exponential_func_2(x_fit, b) if order==2 else exponential_func(x_fit, b)

        plt.plot(x_fit, y_fit, color=color[order])
        plt.scatter(x_data, line_iou[order], label='Data')
        plt.legend()

    plt.show()


def compute_iou(boxes_a, boxes_b):
    boxes_a = torch.from_numpy(boxes_a)
    boxes_b = torch.from_numpy(boxes_b)
    boxes_a, boxes_b = boxes_a.cuda(), boxes_b.cuda()
    iou = boxes_iou3d_gpu(boxes_a, boxes_b)
    iou = iou.cpu().numpy()
    return iou


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


def post_process_scores_and_boxes(dropped_dt_scores, dropped_dt_boxes):
    num_objects = dropped_dt_scores.shape[1]
    mean_dt_scores = np.zeros(11)

    dt_iou = np.zeros((11, num_objects))
    # mean_dt_iou = np.zeros(11)
    for i in range(11):
        mean_dt_scores[i] = np.mean(dropped_dt_scores[i] / dropped_dt_scores[0])
        mean_dt_scores[i] = 1 if mean_dt_scores[i] > 1 else mean_dt_scores[i]
        box_a = dropped_dt_boxes[0].astype(np.float32)
        box_b = dropped_dt_boxes[i].astype(np.float32)
        dt_iou[i] = np.diag(compute_iou(box_a, box_b))
        dt_iou[i][dt_iou[i] > 1] = 1
    mean_dt_iou = np.mean(dt_iou, axis=1)
    return mean_dt_scores, mean_dt_iou


def main(start_idx, end_idx):
    # 11-element lists for plotting
    line_scores, line_iou = [], []
    # 3 order: descent, random, ascent
    for order in range(3):
        mean_dt_scores_multiple, mean_dt_iou_multiple = [], []
        for idx in range(start_idx, end_idx):
            dropped_dt_scores, dropped_dt_boxes = read_dropped_dt_results(idx)
            if len(dropped_dt_scores) == 0:
                continue
            else:
                dropped_dt_scores = dropped_dt_scores[order]
                dropped_dt_boxes = dropped_dt_boxes[order]
                mean_dt_scores, mean_dt_iou = post_process_scores_and_boxes(dropped_dt_scores, dropped_dt_boxes)
                mean_dt_scores_multiple.append(mean_dt_scores)
                mean_dt_iou_multiple.append(mean_dt_iou)
        mean_dt_scores_multiple = np.mean(np.vstack(mean_dt_scores_multiple), axis=0)
        mean_dt_iou_multiple = np.mean(np.vstack(mean_dt_iou_multiple), axis=0)
        line_scores.append(mean_dt_scores_multiple)
        line_iou.append(mean_dt_iou_multiple)

    print(line_scores)
    print(line_iou)
    plot_occam_evaluation(line_scores, line_iou)
    # plot_occam_evaluation_v2(line_scores, line_iou)


if __name__ == '__main__':
    main(0, 501)
