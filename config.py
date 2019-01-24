import os
class Config():
    pass

def get_config():
    train_flag = 'train/'
    cfg = Config()
    cfg.annotation_path = '/home/data/ACDC/Annotation'
    cfg.images_fold_path = '/home/data/ACDC/Images'
    cfg.mask_path = '/home/data/ACDC/train/mask'
    cfg.sample_mask_path = '/home/data/ACDC/train/sample_mask'

    cfg.patch_path = '/home/data/ACDC/train/patch'

    cfg.thumbnail_path = '/home/data/ACDC/train/thumbnail'

    cfg.patch_size = 1024
    cfg.stride = 512
    # cfg.
    for each in [cfg.mask_path, cfg.sample_mask_path, cfg.patch_path, cfg.thumbnail_path]:
        if not os.path.exists(each):
            print(f'make dir => {os.path.abspath(each)}')
            os.mkdir(each)
    return cfg

# train2
# def get_config():
#     cfg = Config()
#     cfg.annotation_path = '/home/data/ACDC/Images_stage2/annotation'
#     cfg.images_fold_path = '/home/data/ACDC/Images_stage2/images'
#     cfg.mask_path = '/home/data/ACDC/Images_stage2/mask'
#     cfg.sample_mask_path = '/home/data/ACDC/Images_stage2/sample_mask'

#     cfg.patch_path = '/home/data/ACDC/Images_stage2/patch'

#     cfg.thumbnail_path = '/home/data/ACDC/Images_stage2/thumbnail'

#     cfg.patch_size = 1024
#     cfg.stride = 512
#     # cfg.
#     for each in [cfg.mask_path, cfg.sample_mask_path, cfg.patch_path, cfg.thumbnail_path]:
#         if not os.path.exists(each):
#             print(f'make dir => {os.path.abspath(each)}')
#             os.mkdir(each)
#     return cfg


# def get_config():
#     cfg = Config()
#     cfg.annotation_path = '/media/steer/data1/Annotation'
#     cfg.images_fold_path = '/media/steer/data1/Images'
#     cfg.mask_path = '/media/steer/data1/mask'
#     cfg.sample_mask_path = '/media/steer/data1/sample_mask'

#     cfg.patch_path = '/media/steer/data1/patch'

#     cfg.thumbnail_path = '/media/steer/data1/thumbnail'

#     cfg.patch_size = 1024
#     cfg.stride = 512
#     # cfg.
#     for each in [cfg.mask_path, cfg.sample_mask_path, cfg.patch_path, cfg.thumbnail_path]:
#         if not os.path.exists(each):
#             print(f'make dir => {os.path.abspath(each)}')
#             os.mkdir(each)
#     return cfg