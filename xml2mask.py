import multiresolutionimageinterface as mir
import os
from os import path
from config import get_config
from glob import glob
import utils 

def main():
    reader = mir.MultiResolutionImageReader()
    cfg = get_config()

    img_fold = cfg.images_fold_path
    anno_path = cfg.annotation_path
    mask_path = cfg.mask_path

    img_fnames = glob(path.join(img_fold, '*.tif'))  # get all tif
    
    for fname in img_fnames:
        img_id = utils.fname_to_id(fname)
        xml_fname = path.join(anno_path, f'{img_id}.xml')

        mr_image = reader.open(fname)
        annotation_list = mir.AnnotationList()
        xml_repository = mir.XmlRepository(annotation_list)
        xml_repository.setSource(xml_fname)
        xml_repository.load()
        annotation_mask = mir.AnnotationToMask()
        output_path = path.join(mask_path, f'{img_id}.tif')
        annotation_mask.convert(annotation_list, output_path,
                                mr_image.getDimensions(), mr_image.getSpacing())


if __name__ == '__main__':
    main()
