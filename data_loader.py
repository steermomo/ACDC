import torch.utils.data as data_utils
from preprocess.config import get_config
from openslide import OpenSlide
from preprocess import utils
from torchvision import transforms
from collections import defaultdict
import numpy as np
import torch
import openslide


class DataProvider(data_utils.Dataset):
    def __init__(self, val=False):
        super(DataProvider, self).__init__()
        self.val = val
        self.cfg = get_config()
        current_stat = np.random.get_state()
        np.random.seed(0)
        img_ids = list(range(1, 151))
        val_ids = np.random.choice(img_ids, 30, replace=False)
        train_ids = list(set(img_ids) - set(val_ids))
        np.random.set_state(current_stat)
        if val:
            self.img_ids = val_ids
        else:
            self.img_ids = train_ids
        self.tif_reader = self._get_reader()
        self.annotaions = self._get_annotations()
        if self.val:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=30 / 255, saturation=0.25, hue=0.04, contrast=0.75),
                    transforms.ToTensor()
                ]
            )

    def _get_reader_by_id(self, img_id):
        """
        某些slide存在空缺,读取会导致 OpenSlide error, 使得无法进行后续读取
        :param img_id:
        :return:
        """
        tif_fp = utils.id_to_fname(img_id)
        reader = OpenSlide(tif_fp)
        return reader

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
        print('@==@')
        return ret

    def __len__(self):
        if self.val:
            return 50000
        return 100000

    def _get_region(self):
        current_label = np.random.randint(2)  # 当前采样label
        label = 1 - current_label
        while label != current_label:
            _tif_idx = np.random.randint(len(self.tif_reader))
            _reader = self.tif_reader[_tif_idx]
            annot_idx = np.random.randint(len(self.annotaions[self.img_ids[_tif_idx]]))
            annot = self.annotaions[self.img_ids[_tif_idx]][annot_idx]
            x, y, w, h, label = annot
            w = self.cfg.patch_size
            h = w
        try:
            row_rand_shift = np.random.randint(8) - 4
            col_rand_shift = np.random.randint(8) - 4
            img = _reader.read_region((y+col_rand_shift, x+row_rand_shift), 0, (w, h)).convert('RGB')
        except openslide.lowlevel.OpenSlideError:
            # print(f'@@@Slide: {self.img_ids[_tif_idx]}, {x, y, w, h}')
            # OpenSlide 对象失效,重新创建
            self.tif_reader[_tif_idx] = self._get_reader_by_id(self.img_ids[_tif_idx])
            return self._get_region()
        return img, label

    def __getitem__(self, idx):
        # current_label = np.random.randint(2)  # 当前采样label
        # label = 1 - current_label
        # while label != current_label:
        #     _tif_idx = np.random.randint(len(self.tif_reader))
        #     _reader = self.tif_reader[_tif_idx]
        #     annot_idx = np.random.randint(len(self.annotaions[self.img_ids[_tif_idx]]))
        #     annot = self.annotaions[self.img_ids[_tif_idx]][annot_idx]
        #     x, y, w, h, label = annot
        # img = _reader.read_region((y, x), 0, (w, h)).convert('RGB')
        img, label = self._get_region()
        img = np.array(img)
        if not self.val:
            img = np.rot90(img, np.random.randint(4))
        img = self.transform(img)
        return img, label


class AllPatchProvider(DataProvider):
    def __init__(self):
        super(AllPatchProvider, self).__init__(val=False)

    def _cal_patch_nums(self):
        cnt = 0
        cnt_level = []  # 记录patch idx 对应的img_id
        for img_id in self.img_ids:
            annotations = self.annotaions[img_id]  # 保证有序
            cnt += len(annotations)
            cnt_level.append(cnt)
        self.patch_cnt = cnt
        self.cnt_level = cnt_level

    def __len__(self):
        return self.patch_cnt

    def __getitem__(self, idx):
        for i in range(len(self.cnt_level)):
            if idx > self.cnt_level[i]:
                continue
            reader = self.tif_reader[i]
            if i == 0:
                last_cnt = 0
            else:
                last_cnt = self.cnt_level[i - 1]
            annotation = self.annotaions[i][idx - last_cnt]
            x, y, w, h, label = annotation
            w = self.cfg.patch_size
            h = w
            img_pil = reader.read_region((y, x), 0, (w, h)).convert('RGB')
            img = np.array(img_pil)
            patch_info = [self.img_ids[i], x, y, w, h, label]
            return img, label, patch_info
