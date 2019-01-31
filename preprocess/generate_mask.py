import multiprocessing as mp
from glob import glob
from os import path

import multiresolutionimageinterface as mir

from preprocess import utils
from preprocess.config import get_config

cfg = get_config()


def gene_mask(each_fp):
    reader = mir.MultiResolutionImageReader()
    img_id = int(path.basename(each_fp).partition('.')[0])

    mr_image = reader.open(each_fp)
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_path = utils.id_to_xml(img_id)
    xml_repository.setSource(xml_path)
    xml_repository.load()
    annotation_mask = mir.AnnotationToMask()
    output_path = path.join(cfg.mask_path, f'{img_id}_mask.tif')
    annotation_mask.convert(annotation_list, output_path,
                            mr_image.getDimensions(), mr_image.getSpacing())


def main():

    image_fnames = glob(path.join(cfg.images_fold_path, '*.tif'))
    with mp.Pool() as p:
        p.map(gene_mask, image_fnames)


if __name__ == '__main__':
    main()
