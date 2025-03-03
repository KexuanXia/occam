"""
    This file generates 2D heat maps for random combination strategy.
"""


import pickle
import cv2
import numpy as np


def adjust_pred_boxes(pred_boxes, image_w, image_h):
    x1, y1, x2, y2 = pred_boxes

    x1 = max(0, min(x1, image_w - 1))
    y1 = max(0, min(y1, image_h - 1))
    x2 = max(0, min(x2, image_w - 1))
    y2 = max(0, min(y2, image_h - 1))

    return np.array([x1, y1, x2, y2])


def mask_image(image, mask):
    masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
              255).astype(np.uint8)
    return masked


def read_kitti_info_val(idx):
    file_path = "/home/xkx/kitti/kitti_infos_trainval.pkl"
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    for item in data:
        if item.get('image_idx') == idx:
            return item
    return IndexError


def read_original_dt_results_2d(idx_str):
    read_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CLOC_2d_detection_results_original/'
                 f'{idx_str}.pkl')
    with open(read_path, 'rb') as file:
        data = pickle.load(file)
    pred_boxes = data["bbox"]
    pred_scores = data["scores"]
    pred_labels = data["label_preds"] + 1

    pred_boxes = pred_boxes.cpu().numpy()
    pred_scores = pred_scores.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()

    print(f"Successfully read 2d original detection results from {read_path}")

    return pred_boxes, pred_labels, pred_scores


def norm_image(image):
    """
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    # print(f"for debug: {np.min(image)}, {np.max(np.min(image), 0)}")
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_cam(image, mask):
    """
    generate CAM image
    :param image: [H,W,C]
    :param mask: [H,W]
    :return: tuple(cam,heatmap)
    """
    # mask to heatmap
    mask = norm_image(mask)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    # merge heatmap to original image
    cam = 0.5 * heatmap + 0.5 * image
    return norm_image(cam), heatmap


def iou(box1, box2):
    if len(box1) == 0 or len(box2) == 0:
        return 0
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    tl = np.vstack([box1[:2], box2[:2]]).max(axis=0)
    br = np.vstack([box1[2:], box2[2:]]).min(axis=0)
    intersection = np.prod(br - tl) * np.all(tl < br).astype(float)
    area1 = np.prod(box1[2:] - box1[:2])
    area2 = np.prod(box2[2:] - box2[:2])
    return intersection / (area1 + area2 - intersection)


# generate combination pairs
def generate_random_combination(it_nr=3000, seed=0):
    arr_1, arr_2 = np.arange(it_nr), np.arange(it_nr)
    np.random.seed(seed)
    np.random.shuffle(arr_1)
    np.random.shuffle(arr_2)
    result = np.column_stack((arr_1, arr_2))
    return result


# generate 2D heat maps
def generate_saliency_map(idx, n_masks=3000,
                          save_figures=False,
                          save_heatmap=False,
                          show=False):
    base_det_boxes, base_det_labels, base_det_scores = read_original_dt_results_2d(str(idx).zfill(6))
    num_of_objects = base_det_boxes.shape[0]
    if num_of_objects == 0:
        heat_map = np.zeros((0, 0, 0))
    else:
        i_path = f'/home/xkx/kitti/training/image_2/{str(idx).zfill(6)}.png'
        image = cv2.imread(i_path)
        image_h, image_w = image.shape[:2]
        mask_path = (f'//media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CasCade_2d_detection_results_3000'
                     f'/{str(idx).zfill(6)}/masks/masks_{n_masks}.pkl')
        with open(mask_path, 'rb') as file:
            masks = pickle.load(file)
        CLOC_2d_detection = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/Random_combination/CLOC_detection_results'
                             f'/{str(idx).zfill(6)}_{n_masks}.pkl')
        with open(CLOC_2d_detection, 'rb') as file:
            preds = pickle.load(file)

        # Initialize heat map
        heat_map = np.zeros((num_of_objects, image_h, image_w), dtype=np.float32)

        mask_index = generate_random_combination(seed=0)[:, 1]

        # Generate heat map for each detected object in the image
        for target_index in range(num_of_objects):
            target_box = base_det_boxes[target_index]
            target_box = adjust_pred_boxes(target_box, image_w, image_h)
            print(f"target_box: {target_box}")

            ious = np.zeros(n_masks)
            for i in range(n_masks):
                mask = masks[mask_index[i]]
                pred = preds[i]
                pred_boxes = pred["bbox"]
                pred_scores = pred["scores"]
                pred_boxes = pred_boxes.cpu().numpy()
                pred_scores = pred_scores.cpu().numpy()

                score = 0
                best_score_value = None
                for box, score_value in zip(pred_boxes, pred_scores):
                    current_score = iou(target_box, box) * score_value
                    if current_score > score:
                        score = current_score
                        ious[i] = current_score / score_value
                        best_score_value = score_value
                heat_map[target_index] += mask * score
            print(f'iou.mean: {np.mean(ious)}')
            print(f'iou.max:{np.max(ious)}')
            print(f'iou.min:{np.min(ious)}')
            image_with_bbox, _ = gen_cam(image, heat_map[target_index])
            # print(f"target: {target_index}")
            # print(f"heatmap.shape: {heat_map[target_index].shape}")
            # print(f"heatmap: {heat_map}")
            left_corner = target_box[:2].astype(np.int32)
            right_corner = target_box[2:].astype(np.int32)
            cv2.rectangle(image_with_bbox, left_corner, right_corner,
                          (0, 0, 255), 5)
            if save_figures:
                save_path = (f"/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/Random_combination/d_rise_heat_map_figures/"
                             f"{str(idx).zfill(6)}_{target_index}_{n_masks}.png")
                cv2.imwrite(save_path, image_with_bbox)
                print(f"{str(idx).zfill(6)}_{target_index}_{n_masks}.png has been saved")

            if show:
                cv2.imshow('image', image_with_bbox)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    if save_heatmap:
        save_path = (f"/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/Random_combination/d_rise_heat_map_data/"
                     f"{str(idx).zfill(6)}_{n_masks}.pkl")
        with open(save_path, 'wb') as output_file:
            pickle.dump(heat_map, output_file)
        print(f"heat map has been saved: {heat_map}")
    return heat_map


# only for visualization and debugging
def visualize_preds(idx, n_masks=3000):
    base_det_boxes, base_det_labels, base_det_scores = read_original_dt_results_2d(str(idx).zfill(6))
    num_of_objects = base_det_boxes.shape[0]
    i_path = f'/home/xkx/kitti/training/image_2/{str(idx).zfill(6)}.png'
    image = cv2.imread(i_path)
    image_h, image_w = image.shape[:2]
    mask_path = (f'//media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CasCade_2d_detection_results_3000'
                 f'/{str(idx).zfill(6)}/masks/masks_{n_masks}.pkl')
    with open(mask_path, 'rb') as file:
        masks = pickle.load(file)
    mask_index = generate_random_combination(seed=0)[:, 1]
    CLOC_2d_detection = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/Random_combination/CLOC_detection_results'
                         f'/{str(idx).zfill(6)}_{n_masks}.pkl')
    with open(CLOC_2d_detection, 'rb') as file:
        preds = pickle.load(file)

    for i in range(n_masks):
        mask = masks[mask_index[i]]
        masked_image = image.copy()
        masked_image = mask_image(masked_image, mask)
        pred = preds[i]
        pred_boxes = pred["bbox"]
        pred_scores = pred["scores"]
        pred_boxes = pred_boxes.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
    # Generate heat map for each detected object in the image
        for pred_box in pred_boxes:
            left_corner = pred_box[:2].astype(np.int32)
            right_corner = pred_box[2:].astype(np.int32)
            cv2.rectangle(masked_image, left_corner, right_corner,
                          (0, 0, 255), 5)

        cv2.imshow('image', masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    for i in range(10, 11):
        generate_saliency_map(i, save_figures=False, save_heatmap=False, show=True)
    # visualize_preds(8)

