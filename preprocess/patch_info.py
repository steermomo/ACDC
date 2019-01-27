from preprocess.config import get_config
from os import path
from glob import glob
# from skimage import io
# import numpy as np
from collections import defaultdict


# import matplotlib.pyplot as plt
# import pickle

# plt.switch_backend('agg')


def main():
    save_fp = 'info.txt'
    cfg = get_config()
    patch_path = cfg.patch_path

    patch_txt_fps = glob(path.join(patch_path, '*.txt'))

    slide_cnt = defaultdict(lambda: defaultdict(int))
    cnt = defaultdict(int)

    for each in patch_txt_fps:
        base_name = path.basename(each)
        img_id = int(base_name.split('.')[0])
        print(f'Reading => {each}')
        with open(each, 'rt', encoding='utf-8') as in_file:
            for line in in_file:
                line_sp = line.strip().split(',')
                label = int(line_sp[-1])
                slide_cnt[img_id][label] += 1
                cnt[label] += 1

    with open(save_fp, 'wt', encoding='utf-8') as out_file:
        out_file.write(f'Total:{cnt[0]+cnt[1]}, Pos:{cnt[1]}, Neg:{cnt[0]}\n')
        out_file.write('=' * 10 + '\n')

        slide_cnt_kv = sorted(slide_cnt.items(), key=lambda key: key[0], reverse=False)

        pos_count, neg_cnt = 0, 0
        for k, v in slide_cnt_kv:
            out_file.write(f'Slide@{k}:{v[1]+v[0]}, Pos:{v[1]}, Neg:{v[0]}\n')
            pos_count += v[1]
            neg_cnt += v[0]
            if k == 100:
                out_file.write(f'Pos_cnt:{pos_count}, Neg_cnt:{neg_cnt}\n')
                out_file.write('=' * 10 + '\n')
                pos_count = 0
                neg_cnt = 0
        out_file.write(f'Pos_cnt:{pos_count}, Neg_cnt:{neg_cnt}\n')
        out_file.write('=' * 10 + '\n')

    print(f'Done!\n, All:{cnt[0]+cnt[1]}, Pos:{cnt[1]}, Neg:{cnt[0]}\n save in {path.abspath(save_fp)}')


if __name__ == '__main__':
    main()
