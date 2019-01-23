import os
from os import path

from openslide import OpenSlide

import numpy as np
import skimage
from skimage import morphology, measure
from skimage.draw import polygon
# from skimage.morphology import watershed
from skimage.filters import sobel
import utils
import xml.etree.ElementTree as ET
from config import get_config
from scipy import ndimage
cfg = get_config()


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


def makemask(img_id, scale_factor, mask_size, mask_loc, xml_path):
    """
    Reads xml file and makes annotation mask for entire slide image
    :param annotation_key: name of the annotation key file
    :param size: size of the whole slide image
    :param xml_path: path to the xml file
    :return: annotation mask
    :return: dictionary of annotation keys and color codes
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
            X_pos = int(each_point.attrib['X']) - mask_loc[0]
            Y_pos = int(each_point.attrib['Y']) - mask_loc[1]
            x_points.append(int(X_pos / scale_factor_row))
            y_points.append(int(Y_pos / scale_factor_col))
        rr, cc = polygon(x_points, y_points)
        mat[rr, cc] = 1
    return mat


def get_anno_bbox(img_id):
    """

    """
    xml_fname = utils.id_to_xml(img_id)
    tree = ET.parse(xml_fname)
    root = tree.getroot()
    annotations = root[0]
    minx, miny, maxx, maxy = 1e10, 1e10, 0, 0
    for each_annotation in annotations:
        for each_point in each_annotation[0]:  # coordinate in coordinates
            X_pos = int(each_point.attrib['X'])
            Y_pos = int(each_point.attrib['Y'])

            minx = min(minx, X_pos)
            miny = min(miny, Y_pos)
            maxx = max(maxx, X_pos)
            maxy = max(maxy, Y_pos)
    return (minx, miny, maxx, maxy)


def save_thumbnail(img, save_path):
    # elevation_map = sobel(coins)
    pass


def get_sample_mask(thumbnail: np.ndarray,):
    """
    
    """
    thumbnail = 1. - thumbnail
    # skimage general examples
    elevation_map = sobel(thumbnail)
    markers = np.zeros_like(thumbnail)
    markers[thumbnail <= 1/255.] = 1
    markers[thumbnail >= 15/255.] = 2
    segmentation = morphology.watershed(elevation_map, markers)
    segmentation = ndimage.binary_fill_holes(segmentation - 1)
    segmentation = morphology.remove_small_objects(segmentation)
    return segmentation


def intersect(bbox1, bbox2):
    lr1, lc1, hr1, hc2 = bbox1
    lr2, lc2, hr2, hc2 = bbox2
    if (lr2 <= lr1 <= hr2 and lc2 <= lc1 <= hc2) or (lr1 <= lr2 <= hr1 and lc1 <= lc2 <= hc2):
        return True
    return False


def noname(img_id, ):
    img_fname = utils.id_to_fname(img_id)
    slide, levels, dims = openwholeslide(img_fname)
    # slide = openslide.OpenSlide(img_fname)
    w, h = dims[7]

    thumb = slide.get_thumbnail((w, h))
    r = h, c = w
    scale_factor_row, scale_factor_col = float(
        dims[0][1]) / r, float(dims[0][0]) / c
    thumb_gray = skimage.color.rgb2gray(np.array(thumb))

    sample_mask = get_sample_mask(thumb_gray)

    mask = thumb_gray != 1  # background is 1, samples was splited by background glass

    mask = morphology.binary_dilation(mask)
    mask = morphology.binary_erosion(mask)
    convx = morphology.convex_hull_object(mask)
    convx = morphology.remove_small_holes(convx)
    convx = morphology.remove_small_objects(convx)

    anno_bbox = get_anno_bbox(img_id)
    label_image = measure.label(convx)
    for region in measure.regionprops(label_image):
        # if intersected, this region is annotated
        minr, minc, maxr, maxc = region.bbox
        source_r, source_c = minr * scale_factor_col, minc * scale_factor_row
        if not intersect(region.bbox, anno_bbox):
            continue
        anno_mask = makemask(img_id, (scale_factor_col, scale_factor_row),
                             (maxr-minr, maxc-minc), (source_r, source_c)
                             )
