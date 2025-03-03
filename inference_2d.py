from mmdet.apis import DetInferencer
import mmcv
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import os
import pickle
from pathlib import Path


def generate_mask(image_size, grid_size, prob_thresh):
    image_w, image_h = image_size
    grid_w, grid_h = grid_size
    cell_w, cell_h = math.ceil(image_w / grid_w), math.ceil(image_h / grid_h)
    up_w, up_h = (grid_w + 1) * cell_w, (grid_h + 1) * cell_h

    mask = (np.random.uniform(0, 1, size=(grid_h, grid_w)) <
            prob_thresh).astype(np.float32)
    mask = cv2.resize(mask, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    offset_w = np.random.randint(0, cell_w)
    offset_h = np.random.randint(0, cell_h)
    mask = mask[offset_h:offset_h + image_h, offset_w:offset_w + image_w]
    return mask


def mask_image(image, mask):
    masked = ((image.astype(np.float32) / 255 * np.dstack([mask] * 3)) *
              255).astype(np.uint8)
    return masked


def main(start_idx, end_idx, nr_it=3000, save_result=False, show=False):
    # Initialize the DetInferencer
    inferencer = DetInferencer(model='cascade-rcnn_r50_fpn_1x_coco',
                               weights='checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth')

    for idx in range(start_idx, end_idx):
        # Prepare the input, set hyperparameter
        detection_type = 'Car'
        detection_label = 2
        image_path = f'/home/xkx/kitti/training/image_2/{str(idx).zfill(6)}.png'
        image = mmcv.imread(image_path)
        image_h, image_w = image.shape[:2]
        np.random.seed(0)
        masks = []
        for i in range(nr_it):
            mask = generate_mask(image_size=(image_w, image_h),
                                 grid_size=(16, 16),
                                 prob_thresh=0.5)
            masked = mask_image(image, mask)
            masks.append(mask)
            if show:
                mmcv.imshow(masked)

            # Perform inference
            prediction_result = inferencer(masked)

            labels = np.array(prediction_result['predictions'][0]['labels'])
            scores = np.array(prediction_result['predictions'][0]['scores'])
            bboxes = np.array(prediction_result['predictions'][0]['bboxes'])

            indices = np.where(labels == detection_label)
            labels = labels[indices]
            scores = np.round(scores[indices], 4)
            bboxes = np.round(bboxes[indices], 2)

            num = bboxes.shape[0]
            if num == 0:
                new_data = []
            else:
                prefix = np.array(['Car', -1, -1, -10])
                suffix = np.array([-1, -1, -1, -1000, -1000, -1000, -10])
                prefix = np.tile(prefix, (num, 1))
                suffix = np.tile(suffix, (num, 1))

                new_data = np.hstack([prefix, bboxes])
                new_data = np.hstack([new_data, suffix])
                new_data = np.column_stack((new_data, scores))

                # print(new_data)

            if save_result:
                folder_path = Path(f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CasCade_2d_detection_results_3000'
                                   f'/{str(idx).zfill(6)}')
                results_path = folder_path / "results"
                masks_path = folder_path / "masks"
                results_path.mkdir(parents=True, exist_ok=True)
                masks_path.mkdir(parents=True, exist_ok=True)

                # save masked 2d detection results
                save_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CasCade_2d_detection_results_3000/'
                             f'{str(idx).zfill(6)}/results/{i}.txt')
                np.savetxt(save_path, new_data, fmt='%s')
                print(f"{i}.txt has been saved")

        if save_result:
            # save corresponding masks
            save_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CasCade_2d_detection_results_3000/'
                         f'{str(idx).zfill(6)}/masks/masks_{nr_it}.pkl')
            with open(save_path, 'wb') as output_file:
                pickle.dump(masks, output_file)
                # print(f"{i} mask has been saved")
        print(f"{str(idx).zfill(6)} masks and masked 2d detection results have been saved")


def inference_original(start_idx, end_idx, save_result=False, show=False):
    # Initialize the DetInferencer
    inferencer = DetInferencer(model='mmdetection/configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py',
                               weights='mmdetection/checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth')

    for idx in range(start_idx, end_idx):
        # Prepare the input, set hyperparameter
        detection_type = 'Car'
        detection_label = 2
        image_path = f'/home/xkx/kitti/training/image_2/{str(idx).zfill(6)}.png'
        image = mmcv.imread(image_path)
        image_h, image_w = image.shape[:2]
        np.random.seed(0)

        # Perform inference
        prediction_result = inferencer(image_path, show=show)

        # print(prediction_result)

        labels = np.array(prediction_result['predictions'][0]['labels'])
        scores = np.array(prediction_result['predictions'][0]['scores'])
        bboxes = np.array(prediction_result['predictions'][0]['bboxes'])

        # print(labels)
        # print(scores)
        # print(bboxes)

        indices = np.where(labels == detection_label)
        labels = labels[indices]
        scores = np.round(scores[indices], 4)
        bboxes = np.round(bboxes[indices], 2)

        # 设置框的颜色和字体
        box_color = (255, 0, 0)  # 红色
        font = cv2.FONT_HERSHEY_SIMPLEX

        for box, score, label in zip(bboxes, scores, labels):
            x1, y1, x2, y2 = box.astype(np.int32)
            # 绘制矩形框
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness=2)
            # 绘制文本: 类别 + 置信度
            text = f"{label}: {score:.2f}"
            cv2.putText(image, text, (x1, y1 - 10), font, 0.5, box_color, thickness=1)

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.show()

        num = bboxes.shape[0]
        if num == 0:
            new_data = []
        else:
            prefix = np.array(['Car', -1, -1, -10])
            suffix = np.array([-1, -1, -1, -1000, -1000, -1000, -10])
            prefix = np.tile(prefix, (num, 1))
            suffix = np.tile(suffix, (num, 1))

            new_data = np.hstack([prefix, bboxes])
            new_data = np.hstack([new_data, suffix])
            new_data = np.column_stack((new_data, scores))

            print(new_data)

        if save_result:
            # save original 2d detection results
            # save_path = (f'/media/xkx/TOSHIBA/KexuanMaTH/kitti/training/CasCade_2d_detection_results_original/'
            #              f'{str(idx).zfill(6)}.txt')
            save_path = f'./{str(idx).zfill(6)}.txt'
            np.savetxt(save_path, new_data, fmt='%s')
            print(f"{str(idx).zfill(6)}.txt has been saved")


if __name__ == '__main__':
    # main(0, 9, save_result=False, show=True)
    inference_original(2, 3, save_result=False, show=False)
