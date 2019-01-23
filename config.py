import os
class Config():
    pass

def get_config():
    cfg = Config()
    cfg.annotation_path = '/media/steer/data1/Annotation'
    cfg.images_fold_path = '/media/steer/data1/Image'
    cfg.mask_path = '/media/steer/data1/mask'
    cfg.sample_mask_path = '/media/steer/data1/sample_mask'

    cfg.patch_path = '/media/steer/data1/patch'

    cfg.thumbnail_path = '/media/steer/data1/thumbnail'

    cfg.patch_size = 256
    cfg.stride = 128
    # cfg.
    for each in [cfg.mask_path, cfg.sample_mask_path, cfg.patch_path, cfg.thumbnail_path]:
        if not os.path.exists(each):
            print(f'make dir => {os.path.abspath(each)}')
            os.mkdir(each)
    return cfg