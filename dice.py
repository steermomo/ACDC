from preprocess import utils

from glob import glob
import os
import numpy as np
from os import path
from skimage import morphology, io, transform

val_mask_path = '/home/data/ACDC/Mask'
pred_fold = './pred'


def mask_map(mask, shape):
    out_arr = np.zeros(shape)
    r, c = mask.shape
    for r_idx in range(r):
        for c_idx in range(c):
            out_arr[r_idx * 8 + 122, c_idx * 8 + 122] = mask[r_idx, c_idx]
    # out_arr = morphology.binary_dilation(out_arr, morphology.disk(8))
    return out_arr


def get_annotated_bbox(img_id: int):
    # tif_fp = utils.id_to_fname(img_id)
    # tif_
    bbox = utils.get_anno_bbox(img_id, scale_factor_col=1, scale_factor_row=1)
    return bbox


def main():
    pred_img_fold = glob(path.join(pred_fold, '*/'))
    dice_info = dict()
    for img_fold in pred_img_fold:
        current_img_id = int(path.basename(img_fold.rstrip(path.sep)))
        mask_fp = utils.id_to_mask_fname(current_img_id)
        mask_reader, levels, dims = utils.openwholeslide(mask_fp)

        anno_bboxes = utils.get_annotated_region_bbox(current_img_id)

        print('anno_bboxes')
        for each in anno_bboxes:
            print(each)


        inter_sum_list = []  # 相交区域的面积
        pred_mask_sum = []  # 预测区域的面积
        gt = []
        pred_numpy_fnames = glob(path.join(img_fold, '*.npy'))
        for each_region in pred_numpy_fnames:

            pred_mask_basename = path.basename(each_region).strip('.npy')
            minr, minc, h, w = list(map(int, pred_mask_basename.split('_')))

            current_bbox = (minr, minc, minr + h, minc + w)

            intersect_flag = False
            for box in anno_bboxes:
                if utils.intersect(current_bbox, box):
                    # 当前区域未被标注, 无法计算dice
                    intersect_flag = True
                    break

            if not intersect_flag:
                # 当前区域未被标注, 无法计算dice
                continue

            print(f'{current_bbox}')
            pred_mask = np.load(each_region)

            print(f'mask shape: {pred_mask.shape}')
            pred_mask = pred_mask >= 0.5

            annotate_mask = np.asarray(mask_reader.read_region((minc, minr), 0, (w, h)).convert('L'))

            annotate_mask = annotate_mask > 0
            print(f'annotate_mask shape: {annotate_mask.shape}')

            # annotate_mask = np.resize(annotate_mask, pred_mask.shape)
            # 在level 0 层次下计算
            # pred_mask = np.resize(pred_mask, annotate_mask.shape)
            current_bbox = list(current_bbox)
            current_bbox.append(current_img_id)
            io.imsave(f'{"_".join(map(str, current_bbox))}mask_source.jpg', (pred_mask * 255).astype(np.uint8))
            pred_mask = mask_map(pred_mask, annotate_mask.shape)

            io.imsave(f'{"_".join(map(str, current_bbox))}mask_map.jpg', (pred_mask * 255).astype(np.uint8))

            pred_mask = morphology.binary_dilation(pred_mask, morphology.disk(8))

            io.imsave(f'{"_".join(map(str, current_bbox))}mask_dilation.jpg', (pred_mask * 255).astype(np.uint8))

            # io.imsave(f'{"_".join(map(str, current_bbox))}.jpg', transform.resize(pred_mask, (1024, 1024)))
            # io.imsave(f'{"_".join(map(str, current_bbox))}.jpg', pred_mask)
            pred_mask_sum.append(np.sum(pred_mask))
            gt.append(np.sum(annotate_mask))

            intersection = np.sum(pred_mask & annotate_mask)
            print(f'intersecton size:{intersection}')

            inter_sum_list.append(intersection)

        # whole_a_mask = np.asarray(mask_reader.read_region((0, 0), levels - 4, dims[-4]).convert('L')) > 0

        # 边长为x倍 对应面积为x^2倍
        # a_mask_sum = np.sum(whole_a_mask) * mask_reader.level_downsamples[-4] ** 2

        dice = 2 * np.sum(inter_sum_list) / (np.sum(pred_mask_sum) + np.sum(gt))
        dice_info[current_img_id] = dice
        print(f'img:{current_img_id}, dice:{dice}')

    avg = 0
    with open('dice_info.txt', 'wt', encoding='utf-8') as out_file:
        for img_id, dice in dice_info.items():
            avg += dice
            out_file.write(f'{img_id}, {dice}\n')
        avg /= len(dice_info)
        out_file.write(f'@=@\navg:{avg}')


if __name__ == '__main__':
    main()
