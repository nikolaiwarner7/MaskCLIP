import os
import shutil
os.chdir('/coc/flash3/nwarner30/MaskCLIP/')

#Select the appropriate directory for 
# 'instance' or 'semantic'
from nwarner_common_utils import SETTING, IMG_DIR, OUT_PREFIX


# Check setting config in nwarner_common_utils

for config in ['train', 'val']:
    full_img_dir = IMG_DIR + '/Inst_Seg_RGBSI_Images/' + config
    WRITTEN_DIR = 'data/coco/image_sets/'
    file_to_write_to = WRITTEN_DIR+'%s_%s.txt' % (OUT_PREFIX, config)
    if os.path.exists(file_to_write_to):
        os.remove(file_to_write_to)
    written_dir_file = open(file_to_write_to, 'w')
    for root, dirs, files in os.walk(full_img_dir, topdown=False):
        for name in files:
            # Replace names with stripping extension
            name = name.replace('.npy','')
            written_dir_file.write(str(name) + os.linesep) 
            print(name)