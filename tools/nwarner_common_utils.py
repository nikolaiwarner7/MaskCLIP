CLASS_SPLITS = {'seen': list(range(1,15+1)),
                'unseen': list(range(16,20+1)),
                'worst_unseen': [16,20], # potted plant, tv monitor
                'all': list(range(1,20+1))}
# Testing

# Parameters for producing Maskclip VOC maps loaded in mmseg/apis/single_gpu test test.py
# CHOOSE ONE OF THESE 3, or none during training:
PRODUCING_MASKCLIP_DATA = False  # Fixed flag for dataloader to store raw segs, only run when producing data
VISUALIZING_TRAINED_MODEL = False
EVALUATE_USING_CLIP = True
EVALUATE_AND_VISUALIZE = False

CLIP_SIM_THRESHOLD_PRESENT = 24.5 # Experiment with

DELETE_DATA = False

PRODUCE_MASKCLIP_MAPS_CONFIG = 'val' # 'train' or 'val'
SUBSAMPLE = 2000 # Number of samples to take for producing our maskclip-heatmap dataset
# 1464 in train, 1449 in val
# Set subsample >>, then it runs whole thing

# Automatically handle split and subdirectories accordingly
if PRODUCE_MASKCLIP_MAPS_CONFIG == 'val':
    # want eval numbers for all, seen and unseen
    #SPLIT = 'unseen'
    SPLIT = 'all'
elif PRODUCE_MASKCLIP_MAPS_CONFIG == 'train':
    SPLIT = 'seen'

OUT_ANNOTATION_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Annotations/'+PRODUCE_MASKCLIP_MAPS_CONFIG # 2DO: should specify folders for train or test
OUT_RGB_S_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Images/' + PRODUCE_MASKCLIP_MAPS_CONFIG
