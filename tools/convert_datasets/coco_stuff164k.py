import argparse
import os.path as osp
import os
import shutil
from functools import partial
from glob import glob

from pycocotools.coco import COCO
import mmcv
import numpy as np
from PIL import Image

COCO_LEN = 123287

clsID_to_trID = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    26: 24,
    27: 25,
    30: 26,
    31: 27,
    32: 28,
    33: 29,
    34: 30,
    35: 31,
    36: 32,
    37: 33,
    38: 34,
    39: 35,
    40: 36,
    41: 37,
    42: 38,
    43: 39,
    45: 40,
    46: 41,
    47: 42,
    48: 43,
    49: 44,
    50: 45,
    51: 46,
    52: 47,
    53: 48,
    54: 49,
    55: 50,
    56: 51,
    57: 52,
    58: 53,
    59: 54,
    60: 55,
    61: 56,
    62: 57,
    63: 58,
    64: 59,
    66: 60,
    69: 61,
    71: 62,
    72: 63,
    73: 64,
    74: 65,
    75: 66,
    76: 67,
    77: 68,
    78: 69,
    79: 70,
    80: 71,
    255: 255
}


def convert_to_trainID(maskpath, out_mask_dir, is_train):
    mask = np.array(Image.open(maskpath))
    mask_copy = mask.copy()
    for clsID, trID in clsID_to_trID.items():
        mask_copy[mask == clsID] = trID
    seg_filename = osp.join(
        out_mask_dir, 'train2017',
        osp.basename(maskpath).split('.')[0] +
        '_labelTrainIds.png') if is_train else osp.join(
            out_mask_dir, 'val2017',
            osp.basename(maskpath).split('.')[0] + '_labelTrainIds.png')
    Image.fromarray(mask_copy).save(seg_filename, 'PNG')


def parse_args():
    parser = argparse.ArgumentParser(
        description=\
        'Convert COCO Stuff 164k annotations to mmsegmentation format')  # noqa
    #parser.add_argument('coco_path', default='/data/coco', help='coco stuff path')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--nproc', default=16, type=int, help='number of process')
    args = parser.parse_args()
    return args

def open_files(img_name, coco):
  img = coco.imgs[img_name]
  annIds = coco.getAnnIds(imgIds=[img['id']])
  anns = coco.loadAnns(annIds)
  num_obj = len(anns)
  print(f'contains: {num_obj} objects')
  return anns



def main():
    args = parse_args()
    coco_path = 'data/coco'
    nproc = args.nproc
    print(os.getcwd())
    out_dir = args.out_dir or coco_path
    print(out_dir)

    out_img_dir = osp.join(out_dir, 'images')
    out_mask_dir = osp.join(out_dir, 'annotations')


    #mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'train2017'))
    #mmcv.mkdir_or_exist(osp.join(out_mask_dir, 'val2017'))

    if out_dir != coco_path:
        shutil.copytree(osp.join(coco_path, 'images'), out_img_dir)

    train_list = glob(osp.join(coco_path, 'images', 'train2017', '*.jpg'))
    train_list = [file for file in train_list if '_labelTrainIds' not in file]
    test_list = glob(osp.join(coco_path, 'images', 'val2017', '*.jpg'))
    test_list = [file for file in test_list if '_labelTrainIds' not in file]
    assert (len(train_list) +
            len(test_list)) == COCO_LEN, 'Wrong length of list {} & {}'.format(
                len(train_list), len(test_list))

    if args.nproc > 1:
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID, out_mask_dir=out_mask_dir, is_train=True),
            train_list,
            nproc=nproc)
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID, out_mask_dir=out_mask_dir, is_train=False),
            test_list,
            nproc=nproc)
    else:
        mmcv.track_progress(
            partial(
                convert_to_trainID, out_mask_dir=out_mask_dir, is_train=True),
            train_list)
        mmcv.track_progress(
            partial(
                convert_to_trainID, out_mask_dir=out_mask_dir, is_train=False),
            test_list)

    print('Done!')


if __name__ == '__main__':
    main()
