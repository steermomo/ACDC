# from time import time
import datetime
import os
from os import path

# import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from openslide import OpenSlide
from scipy import ndimage
from skimage import morphology, measure, color, io, filters, transform
from tqdm import tqdm

from data_loader import PredictPatchLoader
from model import VGG_FCN
from preprocess import utils

# plt.switch_backend('agg')
PATCH_SIZE = 2868
PATCH_SIZE = 2452  # 70
# PATCH_SIZE = 4038  # 128 out
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
    mask = morphology.binary_dilation(mask, morphology.disk(4))
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


class time_sp():
    def __init__(self):
        self._t = datetime.datetime.now()

    def time_span(self, info):
        now = datetime.datetime.now()
        time_delta = now - self._t
        print(f'耗时{time_delta.seconds} : {info}')
        self._t = datetime.datetime.now()


def predict(model: nn.Module, slide_reader: OpenSlide, img_id, bbox, scale_factor):
    """

    :param model:
    :param slide_reader:
    :param img_id:
    :param bbox:
    :param scale_factor:
    :return:
    """

    minr, minc, maxr, maxc = bbox
    scf_row, scf_col = scale_factor
    row_st_origin, col_st_origin = int(minr * scf_row), int(minc * scf_col)
    row_ed_origin, col_ed_origin = int(maxr * scf_row), int(maxc * scf_col)



    row_length, col_length = row_ed_origin - row_st_origin, col_ed_origin - col_st_origin

    if row_length < 500 or col_length < 500:
        return

    # print('read region')
    # print(row_st_origin, col_st_origin)
    # print((col_length + (alpha - 1) * PATCH_STRIDE, row_length + (alpha - 1) * PATCH_STRIDE))
    if (row_length - PATCH_SIZE - (alpha - 1) * SD) % PATCH_STRIDE != 0:
        row_ed_origin += PATCH_STRIDE - (row_length - PATCH_SIZE - (alpha - 1) * SD) % PATCH_STRIDE
    if (col_length - PATCH_SIZE - (alpha - 1) * SD) % PATCH_STRIDE != 0:
        col_ed_origin += PATCH_STRIDE - (col_length - PATCH_SIZE - (alpha - 1) * SD) % PATCH_STRIDE
    # dpt_table
    # whole_dpt = []
    # cm_hot = mpl.cm.get_cmap('bwr')
    cm_hot = mpl.cm.get_cmap('coolwarm')

    patch_loader = data_utils.DataLoader(dataset=PredictPatchLoader(slide_reader,
                                                                    row_st_origin, row_ed_origin,
                                                                    col_st_origin, col_ed_origin,
                                                                    PATCH_SIZE, PATCH_STRIDE,
                                                                    alpha, SD, img_id
                                                                    ),
                                         batch_size=1,
                                         num_workers=3)
    cols = (col_ed_origin - col_st_origin - PATCH_SIZE - (alpha - 1) * SD) // PATCH_STRIDE + 1
    all_dpt = []
    dpt_table_row = []
    t = time_sp()
    for data in patch_loader:
        # t.time_span('转为Tensor')
        # row_idx = idx // cols
        # col_idx = idx % cols
        data.squeeze_()
        opt_table = []
        with torch.no_grad():
            data = data.float()
            for opt_row in range(alpha):
                opt_table.append([])
                for opt_col in range(alpha):
                    current_data = data[opt_row*alpha+opt_col, :, :, :]
                    current_data.unsqueeze_(0)
                    # print(current_data.shape)
                    # t.time_span('Tensor 切片')
                    outputs = model(current_data)
                    opts = nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    opts = opts[0, 1, :, :]
                    # print(opts.shape)
                    opt_table[-1].append(opts)

        # t.time_span('预测耗时')
        dpt_table = merge_opt(opt_table)
        # t.time_span('Merge OPT')
        dpt_table_row.append(dpt_table)

        if len(dpt_table_row) == cols:
            row_img = np.concatenate(dpt_table_row, axis=1)
            all_dpt.append(row_img)
            # for each in all_dpt:
            #     print(each.shape)
            # print(f'all_apt len {len(all_dpt)}')
            dpt_table_row = []

    if len(all_dpt) == 0:
        print(bbox)
        return
    whole_pred = np.concatenate(all_dpt)
    # with open('dpt.dump', 'wb') as outfile:
    #     pickle.dump(whole_pred, outfile)
    htop_map = cm_hot(whole_pred)

    dims = slide_reader.level_dimensions
    down_level = -5
    fr = dims[0][1] / dims[down_level][1]
    fc = dims[0][0] / dims[down_level][0]
    # d = int( PATCH_SIZE + (alpha-1) * PATCH_STRIDE / fc)

    # print(
    #     (col_st_origin , row_st_origin),
    #     len(dims) - 5,
    #     (int((col_ed_origin - col_st_origin) // fc), int((row_ed_origin - row_st_origin) // fr))
    # )
    region_img = slide_reader.read_region(
        (col_st_origin, row_st_origin),
        len(dims) + down_level,
        (int((col_ed_origin - col_st_origin) // fc), int((row_ed_origin - row_st_origin) // fr))

    ).convert('RGB')

    # region_img = thumbnail[minr:maxr, minc:maxc]
    tr, tc = whole_pred.shape

    region_img = np.asarray(region_img)
    region_img = transform.resize(region_img, (tr, tc))
    # region_img = np.resize(region_img, (tr, tc, 3))

    save_image = 0.7 * region_img + 0.3 * color.rgba2rgb(htop_map)
    save_image = save_image.clip(0, 1.0)
    info = '_'.join(map(str, (row_st_origin, col_st_origin, row_ed_origin - row_st_origin, col_ed_origin - col_st_origin)))
    if not path.exists(f'pred/{img_id}'):
        os.mkdir(f'pred/{img_id}')
    np.save(f'pred/{img_id}/{info}.npy', whole_pred)
    io.imsave(f'pred/{img_id}/{info}_gray.bmp', whole_pred)
    io.imsave(f'pred/{img_id}/{info}_region.bmp', region_img)
    io.imsave(f'pred/{img_id}/{info}_heatmap.bmp', htop_map)
    io.imsave(f'pred/{img_id}/{info}.bmp', save_image)



def process(img_id):
    tif_path = utils.id_to_fname(img_id)
    slide_reader, levels, dims = openwholeslide(tif_path)
    w, h = dims[7]

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
    for region_bbox in tqdm(sample_bbox):
        predict(model, slide_reader, img_id, region_bbox, (scale_factor_row, scale_factor_col))


if __name__ == '__main__':
    test_slide = '/run/user/1000/gvfs/sftp:host=192.168.1.101/home/data/ACDC/Images/124.tif'
    test_slide = '/home/data/ACDC/Images/135.tif'
    np.random.seed(0)
    img_ids = list(range(1, 151))
    val_ids = np.random.choice(img_ids, 30, replace=False)
    for each_id in val_ids:
        process(each_id)
