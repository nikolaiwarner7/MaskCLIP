import ipdb
# Parameters for producing Maskclip VOC maps loaded in mmseg/apis/single_gpu test test.py
# CHOOSE ONE OF THESE 3, or none during training:
# Enables proper data processing
PRODUCING_MASKCLIP_DATA = False  # Fixed flag for dataloader to store raw segs, only run when producing data
## Use this setting carefully! Only with PRODUCING_MASKCLIP_DATA
DELETE_DATA = False


TRAINING_RGBS_MODEL = False # Enables handling of photometric distortion
# To train RGBS model in distr fashion, specify launcher
TRAINING_RGBSI_MODEL = True # Enables handling of 5D photometric distortion

SETTING = 'coco-instance' # 'instance' or 'semantic' [segmentation] or # coco-instance or # refcoco


EVALUATE_USING_CLIP = False
EVALUATE_AND_VISUALIZE = False

CLIP_SIM_THRESHOLD_PRESENT = 24.5 # Experiment with


PRODUCE_MASKCLIP_MAPS_CONFIG = 'train' # 'train' or 'val'
# also full_train for COCO with sep directory

#REFCOCO_EXP_CONFIG = 'heatmaps1_posclicks2_negclicks1'
#REFCOCO_EXP_CONFIG = 'heatmaps0_posclicks2_negclicks1'
#REFCOCO_EXP_CONFIG = 'heatmaps1_posclicks2_negclicks0_zss'
REFCOCO_EXP_CONFIG = 'heatmaps0_posclicks2_negclicks0_zss'

# num channels imported conditionally below
# Other configs are listed in refcoco_exp_dict below at the bottom, see gdoc for proc

SUBSAMPLE = 5000 # Set to 2000 for VOC if want to run the whole thing
                # For instance seg set it to 5,000]

# Changes pre-eval class idx name condition
TEMPORARY_7_TO_5_CH_FIX = '010' # or '010' # 2 options from redunant map generated

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
elif SETTING == 'refcoco':
    IMG_DIR = 'data/refcoco/rgbsi_images/'
    OUT_PREFIX = ''

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

CLASSES_TO_ANALYZE_FOR_VIS =  ['bench', 'bicycle', 'teddy bear', 'oven', 'banana']
CLASSES_TO_ANALYZE_IDXS = [COCO_CLASSSES.index(cls) for cls in CLASSES_TO_ANALYZE_FOR_VIS]

COCO_CLASS_DICT_ID_NAME = dict(enumerate(COCO_CLASSSES))
COCO_CLASS_DICT_NAME_ID = {v:k for k,v in COCO_CLASS_DICT_ID_NAME.items()}

VOC_CLASSES = ['airplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair',' cow',
               'dining table', 'dog', 'horse', 'motorcycle', 'person',
               'potted plant', 'sheep', 'couch', 'train', 'tv']

VOC_SEEN_CLASS_IDS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
VOC_UNSEEN_CLASS_IDS  = [15,16,17,18,19]

COCO_SEEN_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                    'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bottle',
                    'chair', 'couch', 'potted plant', 'dining table', 'tv', 
                    ]

COCO_SEEN_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, # 7/20
                        9, 15, 16, 17, 18, 19, 20, 40, #15/20
                        57, 58, 59, 61, 63 # 20/20
                    ] #non zero indexed

COCO_SEEN_CLASS_IDS_ZERO_IDX = [i-1 for i in COCO_SEEN_CLASS_IDS]
COCO_UNSEEN_CLASS_IDS_ZERO_IDX = [i for i in range(80) if i not in COCO_SEEN_CLASS_IDS_ZERO_IDX]

# remove last 5 (potted_plant:tv)
COCO_1565_SEEN_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'boat', 'bird', 'cat', 'dog', 'horse','cow', 'bottle',
                    'chair','dining table'
                    ]

COCO_1565_SEEN_CLASS_IDS = [1, 2, 3, 4, 5, 6,
                        9, 15, 16, 17, 18, 20, 40, #15/20
                        57,  61 ] #non zero indexed


# Aligned with refcoco experiments table
REFCOCO_EXP_DICT = {
    'heatmaps0_posclicks2_negclicks1': {'heatmaps' : 0, 'pos_clickmaps' : 2, 'neg_clickmaps': 1},
    'heatmaps1_posclicks2_negclicks1': {'heatmaps' : 1, 'pos_clickmaps' : 2, 'neg_clickmaps': 1},
    'heatmaps1_posclicks1_negclicks1': {'heatmaps' : 1, 'pos_clickmaps' : 1, 'neg_clickmaps': 1},
    'heatmaps1_posclicks1_negclicks0': {'heatmaps' : 1, 'pos_clickmaps' : 1, 'neg_clickmaps': 0}
}

if REFCOCO_EXP_CONFIG == 'heatmaps1_posclicks2_negclicks1':
    NUM_CHANNELS = 7
elif REFCOCO_EXP_CONFIG == 'heatmaps0_posclicks2_negclicks1':
    NUM_CHANNELS = 6
elif REFCOCO_EXP_CONFIG == 'heatmaps1_posclicks2_negclicks0_zss':
    NUM_CHANNELS = 6
elif REFCOCO_EXP_CONFIG == 'heatmaps0_posclicks2_negclicks0_zss':
    NUM_CHANNELS = 5
else:
    print("error- exp config doesnt correlate to num channels entry in utils")

VOC_PARTS_HUMAN = ['head', 'left eye', 'right eye', 'left ear', 'right ear', 'left eyebrow', 'right eyebrow', 
                    'nose', 'mouth', 'hair', 'torso', 'neck', 'left lower arm', 'let upper arm',
                    'left hand', 'right lower arm', 'right upper arm', 'right hand', 'left lower leg',
                    'left upper leg', 'left foot', 'right lower leg', 'right upper leg', 'right foot']
VOC_PARTS_HUMAN = ['human ' + part for part in VOC_PARTS_HUMAN]

VOC_PARTS_HUMAN_ABBREV = ['head', 'leye', 'reye', 'lear', 'rear', 'lebrow', 'rebrow',
                          'nose', 'mouth', 'hair', 'torso', 'neck', 'llarm', 'luarm',
                          'lhand', 'rlarm', 'ruarm', 'rhand', 'llleg', 'luleg', 'lfoot', 'rlleg',
                          'ruleg', 'rfoot']

###
# Define OpenImages 350 seg classes here, reference elsewhere
import pandas as pd
import os

# opening the file in read mode
import os
print(os.getcwd())
my_file = open("/coc/flash3/nwarner30/MaskCLIP/data/openImages/oidv7-classes-segmentation.txt", "r")
  
# reading the file
data = my_file.read()
  
# replacing end of line('/n') with ' ' and
# splitting the text it further when '.' is seen.
data_into_list = data.split("\n")

# printing the data
#print(data_into_list)
my_file.close()

#contains OI class ids in numeric order as they appear in the txt files for training
modified_list = [item.replace('/', '') for item in data_into_list]
class_id_to_oi_id = {i: val for i, val in enumerate(modified_list)}

# arrange 350 classes alphabetically from open_images
# seen/ unseen split decided later
df_path = '/coc/flash3/nwarner30/MaskCLIP/data/openImages/oidv7-class-descriptions-boxable.csv'
df = pd.read_csv(df_path, sep=',', header=None)
df.head(3)
df_dict = df.to_dict()
new_dict = dict([(value, key) for key, value in df_dict[0].items()])
all_350_oi_seg_classes = [df[1][new_dict[key_val]] for key_val in data_into_list]
#all_350_oi_seg_classes.sort()
ALL_OI_SEG_CLASSES = all_350_oi_seg_classes
ipdb.set_trace()

### Creating seen set from COCO classes for Open Images
# Only 64/80 map from COCO to OI
# Create dictionary for OpenImages(OI):COCO(classes)
OI_COCO_dict = {}

for coco_class in COCO_CLASSSES:
    if coco_class.capitalize() in ALL_OI_SEG_CLASSES:
        OI_COCO_dict[coco_class.capitalize()] = coco_class
    if coco_class.capitalize() not in ALL_OI_SEG_CLASSES:
        #print("not in", coco_class)
        pass

OI_COCO_dict['Motorcycle'] = 'bicycle'
OI_COCO_dict['Aircraft'] = 'airplane'
# no boat class
# no parking meter
# no bench: loveseat (far)
# no cow: cowboy bench (far)
# no umbrella
# no frisbee
# no skis: snowmobile (far)
# no snowboard
# multiple sports balls (football, volleyball, cricket, golf, rugby)
OI_COCO_dict['Ball (Object)'] = 'sports ball'
OI_COCO_dict['Wine'] = 'wine glass'
OI_COCO_dict['Coffee cup'] = 'cup' #also measuring cup
# no fork
OI_COCO_dict['Orange (fruit)'] = 'orange'
OI_COCO_dict['Pastry'] = 'donut'
# no chair
# no potted plant (has squash, lemon as plants)
OI_COCO_dict['Sofa bed'] = 'bed'
# no dining table, has billiard table
OI_COCO_dict['Tablet computer'] = 'tv'
OI_COCO_dict['Remote control'] = 'remote'
OI_COCO_dict['Computer keyboard'] = 'keyboard'
OI_COCO_dict['Mobile phone'] = 'cell phone'
OI_COCO_dict['Microwave oven'] = 'microwave'
# no sink
# no fridge
OI_COCO_dict['Hair dryer'] = 'hair drier'
# no toothbrush

# Map from seen class set to class indexes
# Used to decide what classes to produce in ZSS script
OI_SEEN_CLASSES = []
for seen_cls in OI_COCO_dict:
    OI_SEEN_CLASSES.append(ALL_OI_SEG_CLASSES.index(seen_cls))

OI_UNSEEN_CLASSES = [i for i in list(range(len(ALL_OI_SEG_CLASSES))) if i not in OI_SEEN_CLASSES]

### EXPANDED CLASS SET EXPERIMENT
OI_seen_set = [key for key,val in OI_COCO_dict.items()]
#OI_seen_set
expanded_classes_to_add = [
    'Toy',
    'Radish',
    'Screwdriver',
    'Pretzel',
    'Wrench',
    'Vehicle registration plate',
    'Computer mouse',
    'Coin',
    'Eraser',
    'Sun hat',
    'Human ear',
    'Goldfish',
    'Door handle',
    'Drinking straw',
    'Golf ball',
    'Artichoke',
    'Hair spray',
    'Tea',
    'Wheel',
    'Clothing',
]
expanded_classes_to_remove = [
    'Elephant',
    'Bear',
    'Sheep',
    'Zebra',
    'Giraffe',
    'Truck',
    'Person',
    'Pizza',    
]
# For now just produce additional classes
OI_EXPANDED_SEEN = []
#OI_EXPANDED_SEEN = OI_seen_set.copy()
for cls in expanded_classes_to_add:
    OI_EXPANDED_SEEN.append(cls)
#for cls in expanded_classes_to_remove:
#    OI_EXPANDED_SEEN.remove(cls)
len(OI_EXPANDED_SEEN)

OI_SEEN_EXPANDED_CLASSES = []
for seen_cls in OI_EXPANDED_SEEN:
    OI_SEEN_EXPANDED_CLASSES.append(ALL_OI_SEG_CLASSES.index(seen_cls))

OI_UNSEEN_EXPANDED_CLASSES = [i for i in list(range(len(ALL_OI_SEG_CLASSES))) if i not in OI_SEEN_EXPANDED_CLASSES]

## REDUCED CLASS SET EXPERIMENT

OI_seen_set = [key for key,val in OI_COCO_dict.items()]
#OI_seen_set
""" for ease of experiment, set reduced_classes_add = []
reduced_classes_to_add = [
    'Radish',
    'Screwdriver',
    'Pretzel',
    'Vehicle registration plate',
    'Computer mouse',
     'Human ear',
    'Goldfish',
    'Door handle',
    'Golf ball',
    'Wheel',
    'Clothing',
]
"""
reduced_classes_to_add = []
reduced_classes_to_remove = [
    'Elephant',
    'Bear',
    'Sheep',
    'Zebra',
    'Giraffe',
    'Truck',
    'Person',
    'Pizza',
    'Bus',
    'Cat',
    'Handbag',
    'Suitcase',
    'Surfboard',
    'Tennis racket',
    'Spoon',
    'Banana',
    'Sandwich',
    'Hot dog',
    'Cake',
    'Toaster',
    'Vase',
    'Teddy bear',
    'Wine',
    'Coffee cup',
    'Orange (fruit)',
    'Sofa bed',
    'Tablet computer',
    'Remote control',
    'Microwave oven',
    'Hair dryer',  
]

OI_REDUCED_SEEN = OI_seen_set.copy()
for cls in reduced_classes_to_add:
    OI_REDUCED_SEEN.append(cls)
for cls in reduced_classes_to_remove:
    OI_REDUCED_SEEN.remove(cls)
print("reduced", len( OI_REDUCED_SEEN))   

OI_SEEN_REDUCED_CLASSES = []
for seen_cls in OI_REDUCED_SEEN:
    OI_SEEN_REDUCED_CLASSES.append(ALL_OI_SEG_CLASSES.index(seen_cls))

OI_UNSEEN_REDUCED_CLASSES = [i for i in list(range(len(ALL_OI_SEG_CLASSES))) if i not in OI_SEEN_REDUCED_CLASSES]

## very REDUCED CLASS SET EXPERIMENT

#OI_seen_set = [key for key,val in OI_COCO_dict.items()]
#OI_seen_set
very_reduced_classes_to_add = [
    'Traffic light',
    'Fire hydrant',
    'Bird',
    'Cat',
    'Horse',
    'Backpack',
    'Tie',
    'Kite',
    'Baseball glove',
    'Skateboard',
    'Bottle',
    'Spoon',
    'Carrot',
    'Couch',
    'Laptop',
    'Mouse',
    'Oven',
    'Vase',
    'Scissors',
    'Aircraft',
    'Ball (Object)',
    'Wine',
    'Mobile phone',
]
very_reduced_classes_to_remove = []

OI_VERY_REDUCED_SEEN = []
for cls in very_reduced_classes_to_add:
    OI_VERY_REDUCED_SEEN.append(cls)
for cls in very_reduced_classes_to_remove:
    OI_VERY_REDUCED_SEEN.remove(cls)
print("very reduced", len( OI_VERY_REDUCED_SEEN))    

OI_SEEN_VERY_REDUCED_CLASSES = []
for seen_cls in OI_VERY_REDUCED_SEEN:
    OI_SEEN_VERY_REDUCED_CLASSES.append(ALL_OI_SEG_CLASSES.index(seen_cls))

OI_UNSEEN_VERY_REDUCED_CLASSES = [i for i in list(range(len(ALL_OI_SEG_CLASSES))) if i not in OI_SEEN_VERY_REDUCED_CLASSES]

## very REDUCED CLASS SET EXPERIMENT

#OI_seen_set = [key for key,val in OI_COCO_dict.items()]
#OI_seen_set
extremely_reduced_classes_to_add = [
    'Traffic light',
    'Bird',
    'Horse',
    'Backpack',
    'Tie',
    'Kite',
    'Baseball glove',
    'Skateboard',
    'Bottle',
    'Spoon',
    'Laptop',
    'Mouse',
    'Scissors',
]
extremely_reduced_classes_to_remove = []

OI_EXTREMELY_REDUCED_SEEN = []
for cls in extremely_reduced_classes_to_add:
    OI_EXTREMELY_REDUCED_SEEN.append(cls)

print("extremely reduced", len(OI_EXTREMELY_REDUCED_SEEN))

OI_SEEN_EXTREMELY_REDUCED_CLASSES = []
for seen_cls in OI_EXTREMELY_REDUCED_SEEN:
    OI_SEEN_EXTREMELY_REDUCED_CLASSES.append(ALL_OI_SEG_CLASSES.index(seen_cls))

OI_UNSEEN_EXTREMELY_REDUCED_CLASSES = [i for i in list(range(len(ALL_OI_SEG_CLASSES))) if i not in OI_SEEN_EXTREMELY_REDUCED_CLASSES]


### OI_VOC_DICT for rebuttal

### Creating seen set from COCO classes for Open Images
# Only 64/80 map from COCO to OI
# Create dictionary for OpenImages(OI):COCO(classes)
OI_VOC_dict = {}

for voc_class in VOC_CLASSES:
    if voc_class.capitalize() in ALL_OI_SEG_CLASSES:
        OI_VOC_dict[voc_class.capitalize()] = voc_class
    if voc_class.capitalize() not in ALL_OI_SEG_CLASSES:
        #print("not in", voc_class)
        pass

OI_VOC_dict['Motorcycle'] = 'bicycle'
OI_VOC_dict['Aircraft'] = 'airplane'


# Map from seen class set to class indexes
# Used to decide what classes to produce in ZSS script
OI_VOC_SEEN_CLASSES = []
for seen_cls in OI_VOC_dict:
    OI_VOC_SEEN_CLASSES.append(ALL_OI_SEG_CLASSES.index(seen_cls))

OI_VOC_UNSEEN_CLASSES = [i for i in list(range(len(ALL_OI_SEG_CLASSES))) if i not in OI_VOC_SEEN_CLASSES]
