CLASS_SPLITS = {'seen': list(range(1,15+1)),
                'unseen': list(range(16,20+1)),
                'all': list(range(1,20+1))}


# Parameters for producing Maskclip VOC maps loaded in mmseg/apis/single_gpu test test.py
PRODUCE_MASKCLIP_MAPS_CONFIG = 'train' # 'train' or 'val'
SUBSAMPLE = 11 # Number of samples to take for producing our maskclip-heatmap dataset

# Automatically handle split and subdirectories accordingly
if PRODUCE_MASKCLIP_MAPS_CONFIG == 'val':
    SPLIT = 'unseen'
elif PRODUCE_MASKCLIP_MAPS_CONFIG == 'train':
    SPLIT = 'seen'

OUT_ANNOTATION_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Annotations/'+PRODUCE_MASKCLIP_MAPS_CONFIG # 2DO: should specify folders for train or test
OUT_RGB_S_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Images/' + PRODUCE_MASKCLIP_MAPS_CONFIG