from os import path
from config import get_config

_cfg = get_config()

def id_to_fname(img_id:int ) -> str:
    return ''

def id_to_mask_fname(img_id):
    return ''

def id_to_xml(img_id):
    return ''

def fname_to_id(fname: str) -> int:
    base = path.basename(fname)
    img_id = base.split('.')[0]
    return int(img_id)