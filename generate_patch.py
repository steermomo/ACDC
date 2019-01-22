import os
from os import path

import openslide

import numpy as np
import skimage
from skimage import morphology, measure
import utils
import xml.etree.ElementTree as ET

def get_anno_bbox(img_id):
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

def intersect(bbox1, bbox2):
    lr1, lc1, hr1, hc2 = bbox1
    lr2, lc2, hr2, hc2 = bbox2
    if (lr2 <=lr1 <= hr2 and lc2 <= lc1 <= hc2) or (lr1 <= lr2 <= hr1 and lc1 <= lc2 <= hc2):
        return True
    return False

def noname(img_id, ):
    img_fname = utils.id_to_fname(img_id)
    slide = openslide.OpenSlide(img_fname)
    w, h = slide.level_dimensions[7]
    thumb = slide.get_thumbnail((w, h))
    thumb_gray = skimage.color.rgb2gray(np.array(thumb))
    mask = thumb_gray != 1  # background is 1, samples was splited by background glass

    convx = morphology.convex_hull_object(mask)
    convx = morphology.remove_small_holes(convx)
    convx = morphology.remove_small_objects(convx)

    anno_bbox = get_anno_bbox(img_id)
    label_image = measure.label(convx)
    for region in measure.regionprops(label_image):
        # if intersected, this region is annotated
        minr, minc, maxr, maxc = region.bbox

        