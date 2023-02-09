CLASS_SPLITS = {'seen': list(range(1,15+1)),
                'unseen': list(range(16,20+1)),
                'worst_unseen': [16,20], # potted plant, tv monitor
                'all': list(range(1,20+1))}
# Testing

# Parameters for producing Maskclip VOC maps loaded in mmseg/apis/single_gpu test test.py
# CHOOSE ONE OF THESE 3, or none during training:
# Enables proper data processing
PRODUCING_MASKCLIP_DATA = False  # Fixed flag for dataloader to store raw segs, only run when producing data
## Use this setting carefully! Only with PRODUCING_MASKCLIP_DATA
DELETE_DATA = False

SETTING = 'instance'
TRAINING_RGBS_MODEL = True # Enables handling of photometric distortion
VISUALIZING_TRAINED_MODEL = False
EVALUATE_USING_CLIP = False
EVALUATE_AND_VISUALIZE = False

CLIP_SIM_THRESHOLD_PRESENT = 24.5 # Experiment with


PRODUCE_MASKCLIP_MAPS_CONFIG = 'val' # 'train' or 'val'
SUBSAMPLE = 30 # Number of samples to take for producing our maskclip-heatmap dataset
# 1464 in train, 1449 in val
# Set subsample >>, then it runs whole thing

# Automatically handle split and subdirectories accordingly
if PRODUCE_MASKCLIP_MAPS_CONFIG == 'val':
    # want eval numbers for all, seen and unseen
    #SPLIT = 'unseen'
    SPLIT = 'all'
elif PRODUCE_MASKCLIP_MAPS_CONFIG == 'train':
    SPLIT = 'seen'




###
if SETTING == 'instance':
    IMG_DIR = 'data/VOCdevkit/VOC2012/Inst_Seg_RGBS_Images/'
    OUT_PREFIX = 'RGBS+I'
elif SETTING == 'semantic':
    IMG_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Images/'
    OUT_PREFIX = 'RGBS'

OUT_ANNOTATION_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Annotations/'+PRODUCE_MASKCLIP_MAPS_CONFIG # 2DO: should specify folders for train or test
OUT_RGB_S_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Images/' + PRODUCE_MASKCLIP_MAPS_CONFIG

INST_SEG_OUT_ANNOTATION_DIR = 'data/VOCdevkit/VOC2012/Inst_Seg_RGBS_Annotations/'+PRODUCE_MASKCLIP_MAPS_CONFIG 
INST_SEG_OUT_RGBS_DIR = 'data/VOCdevkit/VOC2012/Inst_Seg_RGBS_Images/' + PRODUCE_MASKCLIP_MAPS_CONFIG
