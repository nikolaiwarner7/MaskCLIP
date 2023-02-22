import os
os.chdir('/coc/flash3/nwarner30/MaskCLIP/')

#Select the appropriate directory for 
# 'instance' or 'semantic'
from nwarner_common_utils import SETTING, IMG_DIR, OUT_PREFIX


# Check setting config in nwarner_common_utils

for config in ['train', 'val']:
    full_img_dir = IMG_DIR + config
    WRITTEN_DIR = 'data/VOCdevkit/VOC2012/ImageSets/Segmentation/'
    written_dir_file = open(WRITTEN_DIR+'%s_%s.txt' % (OUT_PREFIX, config), 'w')
    for root, dirs, files in os.walk(full_img_dir, topdown=False):
        for name in files:
            # Replace names with stripping extension
            name = name.replace('.npy','')
            written_dir_file.write(str(name) + os.linesep) 
            print(name)