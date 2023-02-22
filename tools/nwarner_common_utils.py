
# Parameters for producing Maskclip VOC maps loaded in mmseg/apis/single_gpu test test.py
# CHOOSE ONE OF THESE 3, or none during training:
# Enables proper data processing
PRODUCING_MASKCLIP_DATA = False  # Fixed flag for dataloader to store raw segs, only run when producing data
## Use this setting carefully! Only with PRODUCING_MASKCLIP_DATA
DELETE_DATA = False


TRAINING_RGBS_MODEL = False # Enables handling of photometric distortion
# To train RGBS model in distr fashion, specify launcher
TRAINING_RGBSI_MODEL = True # Enables handling of 5D photometric distortion

SETTING = 'coco-instance' # 'instance' or 'semantic' [segmentation] or # coco-instance


VISUALIZING_TRAINED_MODEL = False
EVALUATE_USING_CLIP = False
EVALUATE_AND_VISUALIZE = False

CLIP_SIM_THRESHOLD_PRESENT = 24.5 # Experiment with


PRODUCE_MASKCLIP_MAPS_CONFIG = 'train' # 'train' or 'val'
# also full_train for COCO with sep directory


SUBSAMPLE = 5000 # Set to 2000 for VOC if want to run the whole thing
                # For instance seg set it to 5,000

######################### DONT MODIFY BELOW ######################### 
######################### #########################  ######################### 
#########################  ######################### ######################### 

# Automatically handle split and subdirectories accordingly
if PRODUCE_MASKCLIP_MAPS_CONFIG == 'val':
    # want eval numbers for all, seen and unseen
    #SPLIT = 'unseen'
    SPLIT = 'all'
elif PRODUCE_MASKCLIP_MAPS_CONFIG == 'train':
    SPLIT = 'seen'

if SETTING == 'instance':
    IMG_DIR = 'data/VOCdevkit/VOC2012/Inst_Seg_RGBS_Images/'
    ANN_DIR  = 'data/VOCdevkit/VOC2012/Inst_Seg_RGBS_Annotations/'
    OUT_PREFIX = 'RGBS+I'
elif SETTING == 'semantic':
    IMG_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Images/'
    ANN_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Annotations/'
    OUT_PREFIX = 'RGBS'
elif SETTING == 'coco-instance':
    IMG_DIR = 'data/coco/images/'
    # ANN_DIR
    OUT_PREFIX = 'RGBS+I'

OUT_ANNOTATION_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Annotations/'+PRODUCE_MASKCLIP_MAPS_CONFIG # 2DO: should specify folders for train or test
OUT_RGB_S_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Images/' + PRODUCE_MASKCLIP_MAPS_CONFIG

INST_SEG_OUT_ANNOTATION_DIR = 'data/VOCdevkit/VOC2012/Inst_Seg_RGBS_Annotations/'+PRODUCE_MASKCLIP_MAPS_CONFIG 
INST_SEG_OUT_RGBS_DIR = 'data/VOCdevkit/VOC2012/Inst_Seg_RGBS_Images/' + PRODUCE_MASKCLIP_MAPS_CONFIG

INST_SEG_COCO_OUT_ANN_DIR = 'data/coco/annotations/Inst_Seg_RGBSI_Annotations/' + PRODUCE_MASKCLIP_MAPS_CONFIG
INST_SEG_COCO_OUT_RGBS_DIR = 'data/coco/images/Inst_Seg_RGBSI_Images/' + PRODUCE_MASKCLIP_MAPS_CONFIG

CLASS_SPLITS = {'seen': list(range(1,15+1)),
                'unseen': list(range(16,20+1)),
                'worst_unseen': [16,20], # potted plant, tv monitor
                'all': list(range(1,20+1))}

COCO_CLASSSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']

COCO_CLASS_DICT_ID_NAME = dict(enumerate(COCO_CLASSSES))
COCO_CLASS_DICT_NAME_ID = {v:k for k,v in COCO_CLASS_DICT_ID_NAME.items()}

COCO_SEEN_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                    'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bottle',
                    'chair', 'couch', 'potted plant', 'dining table', 'tv', 
                    ]
COCO_SEEN_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, # 7/20
                        9, 15, 16, 17, 18, 19, 20, 40, #15/20
                        57, 58, 59, 61, 63 # 20/20
                    ] #non zero indexed