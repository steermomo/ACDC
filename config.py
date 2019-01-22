class Config():
    pass

def get_config():
    cfg = Config()
    cfg.annotation_path = ''
    cfg.images_fold_path = ''
    cfg.mask_path = ''
    cfg.sample_mask_path = ''

    cfg.patch_path = ''
    # cfg.
    return cfg