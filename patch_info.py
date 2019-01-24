from config import get_config
from os import path
from glob import glob
from skimage import io
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
plt.switch_backend('agg')
def main():
    cfg = get_config()
    patch_path = cfg.patch_path
    patch_path = '/home/data/ACDC/train/patch'
    img_fps = glob(path.join(patch_path, '*/*mask.bmp'))
    info = defaultdict(int)
    for each_fp in img_fps:
        img = io.imread(each_fp)
        percent = np.sum(img) / 255 / np.prod(img.shape)
        percent = int(percent * 100)
        info[percent] += 1

    res = []
    for i in range(101):
        res.append(info[i])
    plt.figure(figsize=(16,16))
    plt.plot(res)
    plt.savefig('info.png')
    with open('./info.txt', 'wt', encoding='utf-8') as file:
        for idx, v in enumerate(res):
            print(f'{idx},{v}', file=file)


if __name__ == '__main__':
    main()
    