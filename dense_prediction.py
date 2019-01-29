import os
from os import path
import numpy as np
import skimage
from skimage import morphology, measure, color, io, filters
from model import VGG_FCN
from scipy import ndimage
from glob import glob
import torch
import torch.nn as nn
from openslide import OpenSlide
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.switch_backend('agg')
PATCH_SIZE = 2868
alpha = 4
OPT_SIZE = int((PATCH_SIZE - 244) / 32 + 1)
DPT_SIZE = OPT_SIZE * alpha
# (opt - 1) * 32 + 244
SD = int(32 / alpha)
PATCH_STRIDE = int(SD * DPT_SIZE)


def openwholeslide(path):
    """
    Opens a whole slide image
    :param path: Slide image path.
    :return: slide image, levels, and dimensions
    """

    _directory, _filename = os.path.split(path)
    print('loading {0}'.format(_filename))

    # Open Slide Image
    osr = OpenSlide(path)

    # Get Image Levels and Level Dimensions
    levels = osr.level_count
    dims = osr.level_dimensions
    print('{0} loaded successfully'.format(_filename))
    return osr, levels, dims


def filter_ch(rgb):
    """
    获取slide中字体的mask 取反返回
    注:如47号Slide
    :param rgb: thumbnail 彩图
    :return:
    """
    # (h, w, c) = rgb.shape
    rgb = rgb.astype(np.int)
    r_f = (rgb[:, :, 0] > 47) & (rgb[:, :, 0] < 130)
    g_f = (rgb[:, :, 1] > 47) & (rgb[:, :, 1] < 130)
    b_f = (rgb[:, :, 2] > 40) & (rgb[:, :, 2] < 110)
    mask = (r_f & b_f & g_f)
    mask = morphology.binary_dilation(mask, selem=morphology.disk(3))  # 膨胀 使字体范围扩大
    return ~mask


def get_sample_mask(thumbnail_rgb: np.ndarray):
    """
    获取样本的粗略mask
    update: 2019.01.27 改为 otsu
    :param thumbnail_rgb: 缩略图的彩图
    :return:
    """
    thumbnail_gray = color.rgb2gray(thumbnail_rgb)
    thresh = filters.threshold_otsu(thumbnail_gray)
    mask = thumbnail_gray < thresh

    ch_mask = filter_ch(thumbnail_rgb)

    mask = mask & ch_mask
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask)

    return mask


def overlap1D(xmin1, xmax1, xmin2, xmax2):
    return xmax1 >= xmin2 and xmax2 >= xmin1


def intersect(bbox1, bbox2):
    """
    判定两bbox是否相交
    """
    lr1, lc1, hr1, hc1 = bbox1
    lr2, lc2, hr2, hc2 = bbox2
    # 1 in 2 or 2 in 1
    return overlap1D(lr1, hr1, lr2, hr2) and overlap1D(lc1, hc1, lc2, hc2)


def merge_bbox(bboxes: list):
    """
    整合bbox, 将有相交的整合成更大的bbox
    :param bboxes:
    :return:
    """
    for idx in range(len(bboxes)):
        current_bbox = bboxes[idx]
        for next_idx in range(idx + 1, len(bboxes)):
            other_bbox = bboxes[next_idx]
            if intersect(current_bbox, other_bbox):
                minr = min(current_bbox[0], other_bbox[0])
                minc = min(current_bbox[1], other_bbox[1])
                maxr = max(current_bbox[2], other_bbox[2])
                maxc = max(current_bbox[3], other_bbox[3])
                bboxes.pop(idx)
                bboxes.pop(next_idx - 1)
                bboxes.append((minr, minc, maxr, maxc))
                return merge_bbox(bboxes)
    return bboxes


def merge_opt(opt: list):
    # row_sz, col_sz = opt[0][0].shape
    dpt = np.ones((DPT_SIZE, DPT_SIZE), dtype=np.float64)
    for row in range(DPT_SIZE):
        for col in range(DPT_SIZE):
            i = row % alpha
            j = col % alpha
            r = int(row / alpha)
            c = int(col / alpha)
            dpt[row][col] = opt[i][j][r][c]
    return dpt


def get_region_thumbnail():
    pass


def save_predict():
    pass


def predict(model: nn.Module, slide_reader: OpenSlide, thumbnail, bbox, scale_factor):
    """

    :param model:
    :param slide_reader:
    :param sample_mask:
    :param loc:
    :param scale_factor:
    :return:
    """
    minr, minc, maxr, maxc = bbox
    scf_row, scf_col = scale_factor
    row_st_origin, col_st_origin = int(minr * scf_row), int(minc * scf_col)
    row_ed_origin, col_ed_origin = int(maxr * scf_row), int(maxc * scf_col)

    row_length, col_length = row_ed_origin - row_st_origin, col_ed_origin - col_st_origin

    row_ed_origin += PATCH_STRIDE - (row_length - PATCH_SIZE) % PATCH_STRIDE
    col_ed_origin += PATCH_STRIDE - (col_length - PATCH_SIZE) % PATCH_STRIDE
    # dpt_table
    whole_dpt = []
    for row_idx in tqdm(range(row_st_origin, row_ed_origin + 1, PATCH_STRIDE)):
        dpt_table_row = []
        for col_idx in range(col_st_origin, col_ed_origin + 1, PATCH_STRIDE):
            img = slide_reader.read_region((col_idx, row_idx), 0,
                                           (PATCH_SIZE + alpha * SD, PATCH_SIZE + alpha * SD)).convert('RGB')
            img_np = np.asarray(img) / 255.
            opt_tables = []
            for row_shift in range(0, alpha * SD + 1, SD):
                opt_tables.append([])
                for col_shift in range(0, alpha * SD + 1, SD):
                    img_np = np.asarray(img) / 255.
                    current_patch = img_np[row_shift:row_shift + PATCH_SIZE, col_shift:col_shift + PATCH_SIZE]
                    img_tensor = torch.tensor(current_patch).float()
                    img_tensor = img_tensor.permute(2, 0, 1).cuda()
                    img_tensor.unsqueeze_(0)

                    with torch.no_grad():
                        # 1 * 2 * n * n
                        pred = model(img_tensor)
                        prob_map = nn.functional.softmax(pred, dim=1)
                        prob_map_np = prob_map[0, 1, :, :].cpu().numpy()
                        opt_tables[-1].append(prob_map_np)
            dpt_table = merge_opt(opt_tables)
            dpt_table_row.append(dpt_table)

        row_img = np.concatenate(dpt_table_row, axis=1)
        whole_dpt.append(row_img)
    region_img = thumbnail[minr:maxr, minc:maxc]
    whole_pred = np.concatenate(whole_dpt)
    plt.figure(figsize=(20, 20))
    plt.imshow(region_img, cmap=plt.cm.gray)
    plt.imshow(whole_pred, cmap=plt.cm.gist_heat, alpha=0.4)
    info = ','.join(map(str, bbox))
    plt.savefig(f'pred/{info}.bmp')
    # save_im = (whole_pred*255).astype(np.uint8)

    # io.imsave(f'{info}.bmp', save_im)


def process(tif_path):
    slide_reader, levels, dims = openwholeslide(tif_path)
    w, h = dims[6]

    thumbnail = np.asarray(slide_reader.get_thumbnail((w, h)).convert('RGB'))

    r, c = h, w
    scale_factor_row, scale_factor_col = float(
        dims[0][1]) / r, float(dims[0][0]) / c

    thumb_gray = color.rgb2gray(thumbnail)

    sample_mask = get_sample_mask(thumbnail)

    label_image = measure.label(sample_mask)

    sample_bbox = []
    for region in measure.regionprops(label_image):
        if region.area > 10:
            sample_bbox.append(region.bbox)

    sample_bbox = merge_bbox(sample_bbox)

    model = VGG_FCN()
    model = nn.DataParallel(model)
    resume = '/home/hli/acdc_fcn/model_best.pth.tar'
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(resume, checkpoint['epoch']))

    model = model.cuda()
    model.eval()
    for region_bbox in sample_bbox:
        predict(model, slide_reader, thumbnail, region_bbox, (scale_factor_row, scale_factor_col))


if __name__ == '__main__':
    test_slide = '/run/user/1000/gvfs/sftp:host=192.168.1.101/home/data/ACDC/Images/124.tif'
    test_slide = '/home/data/ACDC/Images/124.tif'
    process(test_slide)
