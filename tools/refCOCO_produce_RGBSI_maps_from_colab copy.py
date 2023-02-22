import numpy as np
import torch
import copy
import json
import cv2
from PIL import Image
from pycocotools.coco import COCO
from refer import REFER
import skimage.io as io
from tqdm import tqdm
import os.path as osp

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Check this based on debug cell for skynet
import matplotlib.pyplot as plt
import shutil
import sys
import mmcv
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

# Main data creation pipeline
# Need to import whether we are doing train or val
# Just to test 
from nwarner_common_utils import PRODUCE_MASKCLIP_MAPS_CONFIG
OUTPUT_IMG_DIR = 'data/coco/images/Inst_Seg_RGBSI_Images'+'/'+PRODUCE_MASKCLIP_MAPS_CONFIG
OUTPUT_ANN_DIR = 'data/coco/annotations/Inst_Seg_RGBSI_Annotations'+'/'+PRODUCE_MASKCLIP_MAPS_CONFIG   

## Set True before running
DELETE_DATA = False

from nwarner_common_utils import COCO_SEEN_CLASS_IDS

os.chdir('/srv/essa-lab/flash3/nwarner30/MaskCLIP/')
sys.path.append('/srv/essa-lab/flash3/nwarner30/MaskCLIP/tools')
sys.path.append('/srv/essa-lab/flash3/nwarner30/MaskCLIP/')


from configs._base_.datasets.coco2017 import img_norm_cfg  
K = 1 #@nik: vary this number to get more or less points
#NUMBER_OF_IMAGES = 100 # how many images to consider in the dataset
NUMBER_OF_IMAGES = None # if use all
# Runs at 5 itr/sec, so each 10,000 images takes 35 min (whole train: 100k, whole val: 5k)
# First N images are selected above
DISTRIBUTED = False


def main():
  dataset = 'refcoco'
  splitBy = 'unc'
  refer = REFER(data_root, dataset, splitBy)


  '''
  Run the data creation pipeline
  '''
  #dataDir = '/cns/oi-d/home/meerahahn/datasets/public_datasets/coco/coco_stuff164k' #@nik: change to your directory
  dataDir = '/srv/essa-lab/flash3/nwarner30/MaskCLIP/data/coco'

  # Makes sure we are accessing the correct data during validation and training
  # Filtered to seen/unseen classes later
  if PRODUCE_MASKCLIP_MAPS_CONFIG == 'train':
    dataType='train2017' #@nik: run for 'val2017' and 'train2017'
  elif PRODUCE_MASKCLIP_MAPS_CONFIG == 'full_train':
    dataType='train2017' #@nik: run for 'val2017' and 'train2017'
  elif PRODUCE_MASKCLIP_MAPS_CONFIG == 'val':
    dataType='val2017'

  print("Building dataset annotations for the following split:", PRODUCE_MASKCLIP_MAPS_CONFIG)


  annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
  coco=COCO(annFile) #@nik: this takes a little bit to load

  ## Load pretrained model to perform CLIP heatmap inference

  cfg_file = 'configs/maskclip/maskclip_r50_512x512_nwarner_coco.py'
  ckpt_file = 'pretrain/RN50_clip_backbone.pth' 
  cfg = mmcv.Config.fromfile(cfg_file)

  cfg.model.train_cfg = None
  model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
  fp16_cfg = cfg.get('fp16', None)
  if fp16_cfg is not None:
      wrap_fp16_model(model)
  checkpoint = load_checkpoint(model, ckpt_file, map_location='cpu')
  dataset = build_dataset(cfg.data.train)

  if 'CLASSES' in checkpoint.get('meta', {}):
      model.CLASSES = checkpoint['meta']['CLASSES']
  else:
      print('"CLASSES" not found in meta, use dataset.CLASSES instead')
      model.CLASSES = dataset.CLASSES

  # Pick the first subset of images accordingly
  img_names = []
  itr = 0
  for img_name in coco.getImgIds():
      img_names.append(img_name)
      itr +=1

      if NUMBER_OF_IMAGES:
        if itr>NUMBER_OF_IMAGES:
          break

  bw_ctr = 0


  model = model.cuda()
  model.eval()
  singleproc_produce_maps(model, coco, dataDir, dataType, bw_ctr)


def singleproc_produce_maps(model, img_names, refer, dataDir, dataType, bw_ctr):
  itr = 0 

  for obj_instance in tqdm(refer.getRefIds()):
    ref = refer.Refs[obj_instance]
    ann_id = ref['ann_id']
    ann = refer.Anns[ann_id]
    test_mask = refer.getMask(ref)

    # get image as np array
    image = refer.Imgs[ref['image_id']]
    img_name = image['file_name'].replace('COCO_train2014_','')
    gt_im = io.imread(osp.join(refer.IMAGE_DIR, img_name))


    if len(gt_im.shape) < 3:
      # Dont load b&w images for now for normalization
      bw_ctr+=1
      print("Bw image!", bw_ctr)
      continue 
    norm_img = mmcv.imnormalize(gt_im, np.array(img_norm_cfg['mean']), np.array(img_norm_cfg['std']), img_norm_cfg['to_rgb'])
    model_batched_input = torch.tensor((np.transpose(gt_im[np.newaxis],(0,3,1,2))/255).astype(np.float32)).cuda()

    result = model.inference(model_batched_input, img_meta=[[{'ori_shape': None}]], rescale=False)

      obj_mask = coco.annToMask(obj_ann)
      #print(obj_mask.shape)
      e_obj_mask = erode(obj_mask, iters=2)
      mask_size = len(np.where(e_obj_mask ==1)[0])
      if mask_size <= K:
        e_obj_mask = obj_mask 
      points = select(obj_mask, obj_ann, min(mask_size, K))
      for point in points:
          #print(point)
          coco_90_idx = point['category_id']
          # coco labels are indexed to 91 categories, only 80 present
          # we only calculate saliency maps for those 80, so map 91-indexed to 80 index using enumeration
          coco_80_cat = [(coco.cats[entry]['name'], i) for i, entry in enumerate(coco.cats) if coco.cats[entry]['id'] == coco_90_idx][0]
          # will show ('tv', 62) for example (0 indexed)
          coco_80_idx = coco_80_cat[1]
          
          #coco_80_idx is 0 idxd, COCO_SEEN_IDs is 1 indexed, so add 1
  

        output_saliency = result[0,coco_80_idx].detach().cpu().numpy()
        output_saliency = output_saliency[:,:,np.newaxis]

        h,w = point['coord']
        output_clickmap = np.zeros(output_saliency.shape)
        output_clickmap[h, w, 0] = 1

        output_shard = np.concatenate((gt_im, output_saliency, output_clickmap),axis=-1)

        out_name = img['file_name'].replace('.jpg','_class%s_inst%s' %(coco_80_idx, point['id']))
        #import os
        #print(os.getcwd())
        np.save(OUTPUT_IMG_DIR+'/'+ out_name, output_shard )

        # Now save the GT fg/bg seg
        obj_mask_batched = obj_mask[:,:,np.newaxis]
        #print(obj_mask_batched.shape)
        np.save(OUTPUT_ANN_DIR+'/'+ out_name, obj_mask_batched)

    if NUMBER_OF_IMAGES:
      itr += 1
      if itr>NUMBER_OF_IMAGES:
        break

if __name__ == '__main__':
    os.chdir('/coc/flash3/nwarner30/MaskCLIP/')

    main()
