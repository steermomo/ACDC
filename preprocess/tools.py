from os import path
from glob import glob
from preprocess.config import get_val_config
import shutil

# cfg = get_config()

cfg = get_val_config()


def merge_anno_file():
    anno_fps = glob(path.join(cfg.patch_path, '*.txt'))
    target_fp = path.join(
        path.join(cfg.patch_path, path.pardir),
        'train_annotation.txt'
    )
    with open(target_fp, 'wt', encoding='utf-8') as out_file:
        for each_fp in anno_fps:
            with open(each_fp, 'rt', encoding='utf-8') as in_file:
                for line in in_file:
                    if line.strip():
                        out_file.write(line)


def val_to_train_fold():
    cfg = get_val_config()
    for fp in ['/home/data/ACDC/val/mask']:
        current_files = glob(path.join(fp, '*'))
        for each_file in current_files:
            base_name = path.basename(each_file)
            img_id, ext = base_name.split('.')
            img_id = int(img_id.split('_')[0]) + 100
            new_base_name = f'{img_id}_mask.{ext}'
            new_fp = path.join(fp, new_base_name)
            print(f'{each_file} ==> {new_fp}')
            shutil.move(each_file, new_fp)


if __name__ == "__main__":
    # merge_anno_file()

    val_to_train_fold()
