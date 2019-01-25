from config import get_config
from skimage import io
from glob import glob
import os
import numpy as np

cfg = get_config()
def center_region(img: np.ndarray):
    """
    
    Arguments:
        img {np.ndarray} -- [description]
    """
    st = int(cfg.patch_size / 2 - cfg.patch_center_sz / 2)
    ed = st + cfg.patch_center_sz
    if np.sum(img[st:ed, st:ed]) > 0:
        return True
    return False


def main():
    mask_fps = glob(os.path.join(
        cfg.patch_path, '*/*mask.bmp'
    ))
    for each_fp in mask_fps:
        im = io.imread(each_fp)
        