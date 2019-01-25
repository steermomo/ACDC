import torch.utils.data as data_utils
from config import get_config, get_val_config
from openslide import OpenSlide
import utils
from glob import glob
from os import path
from torchvision import transforms
from collections import defaultdict
import numpy as np


class DataProvider(data_utils.Dataset):
    def __init__(self, val=False):
        super(DataProvider, self).__init__()

        self.cfg = get_config()
        if val:
            self.img_ids = list(range(100, 151))
        else:
            self.img_ids = list(range(1, 101))
        self.tif_reader = self._get_reader()
        self.annotaions = self._get_annotations()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                
                transforms.ToTensor()
            ]
        )

    def _get_reader(self):
        ret = []
        for each_id in self.img_ids:
            tif_fp = utils.id_to_fname(each_id)
            reader = OpenSlide(tif_fp)
            ret.append(reader)
        return ret

    def _get_annotations(self):
        ret = defaultdict(list)
        for each_id in self.img_ids:
            anno_file = utils.id_to_anno_fname(each_id)
            print(f'read file => {anno_file}')
            with open(anno_file, 'rt', encoding='utf-8') as in_file:
                for line in in_file:
                    line_sp = line.strip().split(',')
                    line_sp = list(map(int, line_sp))
                    img_id, x, y, w, h, label = line_sp
                    ret[each_id].append([x, y, w, h, label])
        return ret

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        _tif_idx = np.random.randint(len(self.tif_reader))
        _reader = self.tif_reader[_tif_idx]
        annot_idx = np.random.randint(len(self.annotaions[self.img_ids[_tif_idx]]))
        annot = self.annotaions[self.img_ids[_tif_idx]][annot_idx]
        x, y, w, h, label = annot
        img = _reader.read_region((y, x), 0, (w, h)).convert('RGB')
        img = np.array(img)
        img = self.transform(img)
        return img, label
