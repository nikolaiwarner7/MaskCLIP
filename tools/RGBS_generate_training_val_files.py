import os
os.chdir('/root/MaskCLIP')

IMG_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Images/'

for config in ['train', 'val']:
    full_img_dir = IMG_DIR + config
    WRITTEN_DIR = 'data/VOCdevkit/VOC2012/ImageSets/Segmentation/'
    written_dir_file = open(WRITTEN_DIR+'RGBS_%s.txt' % config, 'w')
    for root, dirs, files in os.walk(full_img_dir, topdown=False):
        for name in files:
            # Replace names with stripping extension
            name = name.replace('.npy','')
            written_dir_file.write(str(name) + os.linesep) 
            print(name)