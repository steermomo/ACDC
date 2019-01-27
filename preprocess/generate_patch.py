import os
from os import path

from openslide import OpenSlide

import numpy as np
import skimage
from skimage import morphology, measure, color, io, filters
from skimage.draw import polygon
# from skimage.morphology import watershed
from preprocess import utils
import xml.etree.ElementTree as ET
from preprocess.config import get_config
from scipy import ndimage
from glob import glob
import multiprocessing as mp

cfg = get_config()
verbose = True  # 保存缩略图
save_patch_mask = False

if verbose:
    import matplotlib.pyplot as plt

    plt.switch_backend('agg')
    import matplotlib.patches as mpatches


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


def makemask(img_id, scale_factor, mask_size, mask_loc):
    """
    弃用
    """

    xml_fname = utils.id_to_xml(img_id)
    tree = ET.parse(xml_fname)
    root = tree.getroot()
    annotations = root[0]
    scale_factor_row, scale_factor_col = scale_factor
    # Generate annotation array and key dictionary
    mat = np.zeros((mask_size[0], mask_size[1]), dtype='uint8')

    for each_annotation in annotations:
        x_points = []
        y_points = []
        for each_point in each_annotation[0]:  # coordinate in coordinates
            X_pos = float(each_point.attrib['Y']) - mask_loc[0]
            Y_pos = float(each_point.attrib['X']) - mask_loc[1]
            x_points.append(int(X_pos / scale_factor_row))
            y_points.append(int(Y_pos / scale_factor_col))
        rr, cc = polygon(x_points, y_points)
        mat[rr, cc] = 1
    return mat


def get_anno_bbox(img_id, scale_factor_row, scale_factor_col):
    """
    根据xml文件获取标注的整体bbox
    """
    xml_fname = utils.id_to_xml(img_id)
    tree = ET.parse(xml_fname)
    root = tree.getroot()
    annotations = root[0]
    minx, miny, maxx, maxy = 1e10, 1e10, 0, 0
    for each_annotation in annotations:
        for each_point in each_annotation[0]:  # coordinate in coordinates
            X_pos = float(each_point.attrib['Y'])
            Y_pos = float(each_point.attrib['X'])

            X_pos /= scale_factor_row
            Y_pos /= scale_factor_col

            minx = min(minx, X_pos)
            miny = min(miny, Y_pos)
            maxx = max(maxx, X_pos)
            maxy = max(maxy, Y_pos)
    return (minx, miny, maxx, maxy)


def save_thumbnail(img, save_path):
    # elevation_map = sobel(coins)
    pass


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
    # if (lr2 <= lr1 <= hr2 and lc2 <= lc1 <= hc2) or (lr1 <= lr2 <= hr1 and lc1 <= lc2 <= hc1):
    #     return True
    # return False


def check_percent(mask_arr, row, col, sz, percent):
    """

    :param mask_arr: mask数组
    :param row:
    :param col:
    :param sz:
    :param percent: 有效百分比
    :return:
    """
    upper_bound = mask_arr.max()
    area = np.sum(mask_arr[row:row + sz, col:col + sz]) / upper_bound
    if area / (sz ** 2) > percent:
        return True
    return False


def save_patches(img_id: int, patch_name, patch, mask: np.ndarray):
    """保存patch
    
    Arguments:
        img_id {int} -- slide id
        patch_name {str} -- patch name
        patch {PIL.Image} -- patch
        mask {np.ndarray} -- [description]
    """

    save_path = path.join(cfg.patch_path, f'{img_id}')
    if not path.exists(save_path):
        os.mkdir(save_path)
    save_patch_name = path.join(save_path, f'{img_id}_{patch_name}')
    patch.save(f'{save_patch_name}.bmp')
    mask *= 255
    if save_patch_mask:
        skimage.io.imsave(f'{save_patch_name}_mask.bmp', skimage.img_as_uint(mask))
    # mask.save(f'{save_patch_name}_mask.bmp')


def center_region(img: np.ndarray):
    """
    判断mask中央位置是否
    Arguments:
        img {np.ndarray} -- [description] 
    """
    st = int(cfg.patch_size / 2 - cfg.patch_center_sz / 2)
    ed = st + cfg.patch_center_sz
    if np.sum(img[st:ed, st:ed]) > 0:
        return True
    return False


def save_patch_info_line(f, msg_list):
    msg_str = [str(each) for each in msg_list]
    msg_str = ','.join(msg_str)
    f.write(f'{msg_str}\n')


def generate_pacth(img_id: int, anno_mask_reader, sample_mask, scale_factor, loc, blank_percent=0.9,
                   anno_percent=0.1):
    """
    生成patch坐标信息
    :param img_id:
    :param slide:
    :param anno_mask_reader: 标注mask tif reader
    :param sample_mask: 样本的mask
    :param scale_factor:
    :param loc:
    :param blank_percent:
    :param anno_percent:
    :return:
    """

    # patches = defaultdict(list)
    # patch_coord = []
    patch_path = cfg.patch_path  # patch info 保存位置
    save_fp = path.join(patch_path, f'{img_id}.txt')
    out_file = open(save_fp, 'at', encoding='utf-8')

    row, col = loc  # location at level 0
    scale_factor_row, scale_factor_col = scale_factor
    mask_R, mask_C = sample_mask.shape  # 当前采样区域mask的大小
    source_R, source_C = int(mask_R * scale_factor_row), int(mask_C * scale_factor_col)  # 当前采样区域在slide中的大小
    patch_sz_in_mask = int(cfg.patch_size / scale_factor_row)  # patch 在mask缩略图中的大小

    for row_idx in range(row, row + source_R - cfg.stride, cfg.stride):  # 迭代slide row
        for col_idx in range(col, col + source_C - cfg.stride, cfg.stride):

            # 缩略图sample mask, 采样区域的相对位置
            mask_r_idx, mask_c_idx = int((row_idx - row) / scale_factor_row), int((col_idx - col) / scale_factor_col)

            # 　非组织部分
            if not check_percent(sample_mask, mask_r_idx, mask_c_idx, patch_sz_in_mask, blank_percent):
                continue

            # !!!read_region 方法 注意传入位置
            # 读取mask tif
            anno_mask = anno_mask_reader.read_region((col_idx, row_idx), 0, (cfg.patch_size, cfg.patch_size))
            anno_mask = color.rgb2gray(np.array(anno_mask))  # RGBA to GRAY

            if center_region(anno_mask):
                label = 1
            else:
                label = 0

            save_patch_info_line(out_file, [img_id, row_idx, col_idx, cfg.patch_size, cfg.patch_size, label])
            # patch_coord.append([
            #
            # ])
    out_file.close()
    print(f'==>{img_id}')
    # return patch_coord


def get_glass_mask(thumbnail_gray: np.ndarray):
    """
    获取玻璃盖板的mask, 用于判断相似组织是否被标记
    注:比赛中相似组织可能只标记一份
    :param thumbnail_gray:
    :return:
    """
    # print(thumbnail_gray.shape)
    # assert thumbnail_gray.shape[-1] == 1
    mask = thumbnail_gray != 1  # background is 1, samples was splited by background glass

    mask = morphology.binary_dilation(mask)
    mask = morphology.binary_erosion(mask)
    convex = morphology.convex_hull_object(mask)
    # convex = morphology.remove_small_holes(convex)
    convex = morphology.remove_small_objects(convex)
    return convex


def process(img_id, ):
    img_fname = utils.id_to_fname(img_id)

    slide_reader, levels, dims = openwholeslide(img_fname)  # slide reader
    anno_mask_reader, _, _ = openwholeslide(utils.id_to_mask_fname(img_id))  # mask reader
    # slide = openslide.OpenSlide(img_fname)
    w, h = dims[6]

    thumb = np.asarray(slide_reader.get_thumbnail((w, h)).convert('RGB'))
    r, c = h, w
    scale_factor_row, scale_factor_col = float(
        dims[0][1]) / r, float(dims[0][0]) / c

    thumb_gray = color.rgb2gray(thumb)

    sample_mask = get_sample_mask(thumb)
    # if verbose:
    #     io.imsave(path.join(cfg.thumbnail_path, f'{img_id}_sample_mask.bmp'), skimage.img_as_ubyte(sample_mask))

    glass_mask = get_glass_mask(thumb_gray)  # 玻璃基板mask

    anno_bbox = get_anno_bbox(img_id, scale_factor_row, scale_factor_row)

    label_image = measure.label(glass_mask)

    if verbose:
        anno_mask_gray = skimage.color.rgb2gray(np.array(anno_mask_reader.get_thumbnail((w, h))))
        anno_mask_gray = anno_mask_gray > 0
        fig, ax = plt.subplots(1, 4, figsize=(24, 16))
        ax[0].imshow(sample_mask, cmap=plt.cm.gray)
        ax[1].imshow(sample_mask * thumb_gray, cmap=plt.cm.gray)
        ax[2].imshow(anno_mask_gray*thumb_gray, cmap=plt.cm.gray)
        ax[3].imshow(thumb_gray, cmap=plt.cm.gray)

        plt.savefig(path.join(cfg.thumbnail_path, f'{img_id}.png'))
        fig, ax = plt.subplots(figsize=(6, 10))
        ax.imshow(label_image)

    for region in measure.regionprops(label_image):
        # if intersected, this region is annotated
        minr, minc, maxr, maxc = region.bbox
        if not intersect(region.bbox, anno_bbox):
            # 当前组织片未被标记
            continue
        if verbose:
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

        source_r, source_c = int(minr * scale_factor_col), int(minc * scale_factor_row)  # region 在slide中的位置
        sample_mask_sub = skimage.img_as_ubyte(sample_mask[minr:maxr, minc:maxc])  # 对应 region 的组织 mask

        generate_pacth(img_id, anno_mask_reader, sample_mask_sub,
                       (scale_factor_row, scale_factor_row), (source_r, source_c))

    if verbose:
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(path.join(cfg.thumbnail_path,
                              f'{img_id}_label_bbox.png'))


def main(white_list=None, black_list=None):
    img_fps = glob(path.join(cfg.images_fold_path, '*.tif'))
    img_ids = [int(path.basename(each_fp).partition('.')[0])
               for each_fp in img_fps]
    if white_list is not None:
        img_ids = [ids for ids in white_list if ids in img_ids]
    if black_list is not None:
        img_ids = [ids for ids in img_ids if ids not in black_list]
    with mp.Pool() as p:
        p.map(process, img_ids)


if __name__ == '__main__':
    # black_str = '64 81 35 67 41 89 36 62 66 38 82 45 61 44 42 72 92 91 98 96 21 27 32 49 28 47'
    # black_list = [int(each) for each in black_str.split(' ')]
    # main(None, black_list)
    main()
    # main([47, 54])
    # process(47)
