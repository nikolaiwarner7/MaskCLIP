import os
import ipdb
VISUALIZING_TRAINED_MODEL = True # Now just used for eval script
# Set to None, or the number of desired visualizations
SAVE_VISUALIZATONS = False # Runs into output/trained_model_vis folders
#VISUALIZING_NUM_SAMPLES = 1e3 # set to 100e3 to use all data 
VISUALIZING_NUM_SAMPLES = 100e3 # set to 100e3 to use all data 

# For camerra ready stuff
OPEN_INFERENCE = True
#OPEN_INFERENCE_QUERY = 'Wrist carpal bones'

# USE FOR VIS TOO
TRAINING_OPEN_IMAGES = False

# For other analysis
NUM_CLASS_VS_IOU_ANALYSIS = False

BOUNDARY_IOU = False
COCO_MVAL_DATAPROC = False #changes name processing    

# logic follows dont modify
if TRAINING_OPEN_IMAGES:
    SETTING = 'open_images'
elif not TRAINING_OPEN_IMAGES:
    SETTING = 'coco'

#SETTING = 'coco'
#SETTING = 'open_images' # see below, sets fair eval crit # or 'open_images'




#########################
from nwarner_common_utils import OI_UNSEEN_CLASSES, OI_SEEN_CLASSES
from nwarner_common_utils import COCO_SEEN_CLASS_IDS_ZERO_IDX, COCO_UNSEEN_CLASS_IDS_ZERO_IDX
from nwarner_common_utils import VOC_SEEN_CLASS_IDS, VOC_UNSEEN_CLASS_IDS
from nwarner_common_utils import class_id_to_oi_id

oi_id_to_class_id = {v: k for k, v in class_id_to_oi_id.items()}
#ipdb.set_trace()

if SETTING == 'open_images':
    SEEN_CLASSES = OI_SEEN_CLASSES
    UNSEEN_CLASSES = OI_UNSEEN_CLASSES
elif SETTING == 'coco' or SETTING == 'refcoco':
    SEEN_CLASSES = COCO_SEEN_CLASS_IDS_ZERO_IDX
    UNSEEN_CLASSES = COCO_UNSEEN_CLASS_IDS_ZERO_IDX
elif SETTING == 'voc':
    SEEN_CLASSES = VOC_SEEN_CLASS_IDS
    UNSEEN_CLASSES = VOC_UNSEEN_CLASS_IDS
else:
    print("ERROR")
print("SETTING", SETTING)

##### Analysis for number of distractors present per image/ mIoU
full_val_path = '/srv/essa-lab/flash3/nwarner30/MaskCLIP/data/openImages/image_sets/110_filtered_val.txt'

my_file = open(full_val_path, "r")
data = my_file.read()
val_files = data.split("\n")

all_val_files = []
anns_per_img = {}
unseen_anns_per_img = {}

for file in val_files:
    # grab object instance
    ann_name = file
    all_val_files.append(ann_name)
    # name format is imgid_classid_boxid
    img_name = ann_name.split('_')[0]
    oi_id = ann_name.split('_')[1]  # Assuming class_id is the second part
    #ipdb.set_trace()
    cls_id = oi_id_to_class_id.get(oi_id, None)

    #img and cls name
    last_bit = '_' + ann_name.split('_')[-1]
    img_cls_name = ann_name.replace(last_bit, '')
    if img_cls_name not in anns_per_img.keys():
        anns_per_img[img_cls_name] = 0
    anns_per_img[img_cls_name]+=1

    if cls_id in UNSEEN_CLASSES:
        if img_cls_name not in unseen_anns_per_img.keys():
            unseen_anns_per_img[img_cls_name] = 0
        unseen_anns_per_img[img_cls_name] += 1
    
# Calculate imgs_per_ann_unseen
imgs_per_ann_unseen = []

# Now loop through again, and put each in list ha
my_file = open(full_val_path, "r")
data = my_file.read()
val_files = data.split("\n")

imgs_per_ann = []
for file in val_files:
    oi_id = file.split('_')[1]  # Assuming class_id is the second part
    #ipdb.set_trace()
    cls_id = oi_id_to_class_id.get(oi_id, None)

    # grab object instance
    ann_name = file
    all_val_files.append(ann_name)
    # name format is imgid_classid_boxid
    last_bit = '_' + ann_name.split('_')[-1]
    img_cls_name = ann_name.replace(last_bit, '')
    #ipdb.set_trace()
    if cls_id in UNSEEN_CLASSES:
        imgs_per_ann_unseen.append(unseen_anns_per_img.get(img_cls_name, 0))
    imgs_per_ann.append(anns_per_img[img_cls_name])

IMGS_PER_ANN = imgs_per_ann
IMGS_PER_ANN_UNSEEN = imgs_per_ann_unseen
#ipdb.set_trace()

#from prompt engineering
PROMPT_TEMPLATES = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.',
]

# for softmax operator
BG_CLASSES = ['building', 'ground', 'grass', 'tree', 'sky']