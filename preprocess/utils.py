import xml.etree.ElementTree as ET
from os import path
import os

import numpy as np
from scipy import ndimage
from skimage import morphology, measure, color, filters
from openslide import OpenSlide

from preprocess.config import get_config

_cfg = get_config()


def id_to_fname(img_id: int) -> str:
    return path.join(_cfg.images_fold_path, f'{img_id}.tif')


def id_to_mask_fname(img_id):
    return path.join(_cfg.mask_path, f'{img_id}_mask.tif')


def id_to_xml(img_id):
    return path.join(_cfg.annotation_path, f'{img_id}.xml')


def id_to_anno_fname(img_id: int) -> str:
    return path.join(_cfg.patch_path, f'{img_id}.txt')


def fname_to_id(fname: str) -> int:
    base = path.basename(fname)
    img_id = base.split('.')[0]
    return int(img_id)


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


def get_anno_bbox(img_id, scale_factor_row, scale_factor_col):
    """
    根据xml文件获取标注的整体bbox
    """
    xml_fname = id_to_xml(img_id)
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


def get_annotated_region_bbox(img_id):
    """
    获取标注区域的bbox
    :param glass_mask:
    :param anno_bbox:
    :return:
    """
    tif_fp = id_to_fname(img_id)
    tif_reader, _, dims = openwholeslide(tif_fp)
    w, h = dims[6]
    thumb =  np.asarray(tif_reader.get_thumbnail((w, h)).convert('RGB'))

    thumb_gray = color.rgb2gray(thumb)

    glass_mask = get_glass_mask(thumb_gray)

    anno_bbox = get_anno_bbox(img_id, scale_factor_row=1, scale_factor_col=1)

    ret = []
    label_image = measure.label(glass_mask)
    for region in measure.regionprops(label_image):
        bbox = list(map(lambda x: x*tif_reader.level_downsamples[6], region.bbox))
        if not intersect(bbox, anno_bbox):
            # 当前组织片未被标记
            continue
        ret.append(bbox)
    return ret
