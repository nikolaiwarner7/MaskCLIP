""" Loads Maskclip iunference module try it on novel class set
Then just run inference through trained model.
"""
# Add these imports at the beginning of your script
import time
import os
import ipdb
import sys
aux_mmseg = '/srv/essa-lab/flash3/nwarner30/miniconda3/envs/maskclip_env/lib/python3.9/site-packages/mmsegmentation-0.20.2-py3.9.egg'
if aux_mmseg in sys.path:
    sys.path.remove(aux_mmseg)


sys.path.append('/coc/flash3/nwarner30/MaskCLIP')
sys.path.append('/coc/flash3/nwarner30/MaskCLIP/mmseg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn as nn
import mmcv
#from mmcv import Config
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from importlib import reload
from mmseg.datasets import build_dataset
import open_inference_config
from open_inference_config import OPEN_INFERENCE_QUERY

import logging

# Suppress mmcv logging messages
logging.getLogger('mmcv').setLevel(logging.WARNING)

# Force sys to use local mmseg for debugging/ smooth compatibility vs pylib one
# Update: it worked

os.chdir('/srv/essa-lab/flash3/nwarner30/MaskCLIP')
# Initialize global variables


# camera_ready_open_vocab_modular.py

# Query will update before each cls seg runs

def load_maskclip_model(cfg_file, ckpt_file):
    cfg = mmcv.Config.fromfile(cfg_file)
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, ckpt_file, map_location='cpu')
    model = model.cuda()
    model.eval()
    return model, checkpoint

def load_seg_model(cfg_file, ckpt_file, num_channels):
    NUM_CHANNELS = 5
    RGBS_NUM_CLASSES = 2 # background + foreground



    cfg = mmcv.Config.fromfile(cfg_file)
    del cfg.model['pretrained']
    cfg.model['backbone'] = {'type': 'ResNetV1c', 'depth': 50, 'num_stages': 4, \
        'out_indices': (0, 1, 2, 3), 'dilations': (1, 1, 2, 4), 'strides': (1, 2, 1, 1),\
                'norm_cfg': {'type': 'SyncBN', 'requires_grad': True}, 'norm_eval': False,\
                    'style': 'pytorch', 'contract_dilation': True, 'pretrained': 'open-mmlab://resnet50_v1c'}

    cfg.model['decode_head'] = {'type': 'DepthwiseSeparableASPPHead', 'in_channels': 2048,\
        'in_index': 3, 'channels': 512, 'dilations': (1, 12, 24, 36), 'c1_in_channels': 256,\
                'c1_channels': 48, 'dropout_ratio': 0.1, 'num_classes': RGBS_NUM_CLASSES,\
                    'norm_cfg': {'type': 'SyncBN', 'requires_grad': True}, \
                    'align_corners': False, 'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}}

    cfg.model['auxiliary_head'] = {'type': 'FCNHead', 'in_channels': 1024, 'in_index': 2, 'channels': 256, 'num_convs': 1,\
            'concat_input': False, 'dropout_ratio': 0.1, 'num_classes': RGBS_NUM_CLASSES, 'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},\
                'align_corners': False, 'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 0.4}}

    ##
    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model.backbone.stem[0] = nn.Conv2d(NUM_CHANNELS, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1))
    checkpoint = load_checkpoint(model, ckpt_file, map_location='cpu')
    model.cpu()
    model.eval()
    return model

def resize_image_to_array(input_path, max_dimension=1024):
    # Open the image file
    img = Image.open(input_path)
    
    # Get the original dimensions
    width, height = img.size
    print("Original image dimensions, ", height, "__", width)
    
    # Check if any dimension is greater than 1300
    if width > 1300 or height > 1300:
        # Calculate the aspect ratio
        aspect_ratio = width / height
        
        # Calculate new dimensions that maintain the original aspect ratio
        if width > height:
            new_width = max_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_dimension
            new_width = int(new_height * aspect_ratio)
            
        # Resize the image
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        
    # Convert the (possibly resized) image to a NumPy array
    img_array = np.array(img)
    
    return img_array
def prepare_input_image(image_path):
    img_array = resize_image_to_array(image_path)
    imh, imw, _ = img_array.shape
    model_input = np.zeros((imh, imw, 5))
    model_input[:, :, :3] = img_array
    return model_input


def prepare_clickmaps(click_points_list, imh, imw):
    dist_clickmaps = []
    for clickh, clickw in click_points_list:
        output_clickmap = np.zeros((imh, imw, 1))
        output_clickmap[clickh, clickw, 0] = 1
        inverse_clickmap = np.logical_not(output_clickmap)
        dist_clickmap = distance_transform_edt(inverse_clickmap)
        dist_clickmap = (dist_clickmap - np.min(dist_clickmap)) / (np.max(dist_clickmap) - np.min(dist_clickmap))
        dist_clickmaps.append(dist_clickmap)
    return dist_clickmaps


def run_inference_maskclip(model, model_input):
    from configs._base_.datasets.coco2017 import img_norm_cfg
    img_norm_cfg['mean'] = [123.675, 116.28, 103.53]
    img_norm_cfg['std'] = [58.395, 57.12, 57.375] 
    model_input[:,:,:3] = mmcv.imnormalize(model_input[:,:,:3], np.array(img_norm_cfg['mean']), np.array(img_norm_cfg['std']), img_norm_cfg['to_rgb']) 
    model_input = model_input[:,:,:3]
    permute_test_img = torch.tensor(model_input[np.newaxis,:,:,:]).permute(0, 3, 1, 2).cuda().float()
    #ipdb.set_trace()
    model.cuda()
    model.float()
    result = model.inference(permute_test_img, img_meta=[{'ori_shape': None}], rescale=False)
    return result


def run_inference_seg(model, maskclip_result, dist_clickmap, seg_model_input):
    output_saliency = maskclip_result
    output_saliency = (output_saliency - np.min(output_saliency))/ (np.max(output_saliency)- np.min(output_saliency))

    #ipdb.set_trace()
    seg_model_input[:,:,-2] = output_saliency
    seg_model_input[:,:,-1] = dist_clickmap[:,:,0]
    
    from configs._base_.datasets.coco2017 import img_norm_cfg
    img_norm_cfg['mean'] = [123.675, 116.28, 103.53, 0.5]
    img_norm_cfg['std'] = [58.395, 57.12, 57.375, 0.5]
    seg_model_input[:,:,:4] = mmcv.imnormalize(seg_model_input[:,:,:4], np.array(img_norm_cfg['mean']), np.array(img_norm_cfg['std']), img_norm_cfg['to_rgb']) 

    model = model.cuda()
    permute_test_img = torch.tensor(seg_model_input[np.newaxis,:,:,:]).permute(0, 3, 1, 2).cuda().float()
    result = model.inference(permute_test_img, img_meta=[{'ori_shape': None}], rescale=False)
    return result


def plot_results(model_input_display_copy, seg_model_input_list, predictions, click_points_list, outname):
    def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
        return mycmap

    cmap1 = transparent_cmap(matplotlib.colors.ListedColormap(['black'] * 255 + ['deepskyblue']))
    cmap2 = transparent_cmap(matplotlib.colors.ListedColormap(['black'] * 255 + ['orange']))

    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5))
    ax1, ax2 = axes1

    ax1.imshow(model_input_display_copy[:, :, :3] / 255)
    for i, click_point in enumerate(click_points_list):
        h, w = click_point
        color = "deepskyblue" if i == 0 else "orange"
        ax1.plot(w, h, marker="o", markersize=20, markeredgecolor="black", markerfacecolor=color, linewidth=5)
    ax1.axis('off')

    ax2.imshow(model_input_display_copy[:, :, :3] / 255)
    for i, prediction in enumerate(predictions):
        cmap = cmap1 if i == 0 else cmap2
        ax2.imshow(prediction, cmap=cmap, alpha=1.0)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(f'output_images/11_20_{outname}_image_and_segmentation.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    fig2, axes2 = plt.subplots(1, len(OPEN_INFERENCE_QUERY), figsize=(10, 5))
    for i, ax in enumerate(axes2):
        ax.imshow(seg_model_input_list[i][:, :, 3])
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'output_images/11_20_{outname}_heatmaps.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def image_to_segmentation(image_name, click_points_list, outname, maskclip_model, seg_model):
    start_time = time.time()
    model_input = prepare_input_image(image_name)

    model_input_display_copy = model_input.copy()
    seg_model_input_list = [model_input.copy() for _ in OPEN_INFERENCE_QUERY]

    dist_clickmaps = prepare_clickmaps(click_points_list, *model_input.shape[:2])
    
    maskclip_result = run_inference_maskclip(maskclip_model, model_input)

    predictions = []
    for i, query in enumerate(OPEN_INFERENCE_QUERY):
        output_saliency = maskclip_result[0, i].detach().cpu().numpy()
        seg_model_input_list[i][:,:,-2] = output_saliency

        seg_result = run_inference_seg(seg_model, output_saliency, dist_clickmaps[i], seg_model_input_list[i])
        prediction = np.argmax(seg_result[0].detach().cpu().numpy(), axis=0)
        predictions.append(prediction)

    plot_results(model_input_display_copy, seg_model_input_list, predictions, click_points_list, outname)
    print("Inference finished")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")

if not os.path.exists('output_images'):
    os.makedirs('output_images')

OUT_NAME = f"{OPEN_INFERENCE_QUERY[0]}_{OPEN_INFERENCE_QUERY[1]}_{time.strftime('%Y_%m_%d')}"
#INFERENCE_IMAGE_NAME = 'batman_joker.jpg'
#click_points_list = [(390, 200), (300, 800)]
#INFERENCE_IMAGE_NAME = 'globe_coffee.jpg'
#click_points_list = [(800, 300), (600, 600)]
#INFERENCE_IMAGE_NAME = 'horse_telephone.jpg'
#click_points_list = [(500, 200), (670, 1080)]
#INFERENCE_IMAGE_NAME = 'magic_wand_tie.jpg'
#click_points_list = [(500, 370), (550, 800)]
#INFERENCE_IMAGE_NAME = 'speaker_casette.jpg'
#click_points_list = [(750, 390), (620, 500)]
#INFERENCE_IMAGE_NAME = 'hammer_drill.jpg'
#click_points_list = [(370, 500), (700, 250)]


#OPEN_INFERENCE_QUERY = ['Basket', 'Model globe']
INFERENCE_IMAGE_NAME = 'globe_candle.jpg'
click_points_list = [(500, 200), (450, 830)]
INFERENCE_IMAGE_NAME = 'lifejacket_kayak2.jpg'
click_points_list = [(280, 200), (300, 550)]
INFERENCE_IMAGE_NAME = 'microscope_hairnet.jpg'
click_points_list = [(570, 920), (140, 510)]

start_time = time.time() 
cfg_file_maskclip = '/srv/essa-lab/flash3/nwarner30/MaskCLIP/configs/maskclip/maskclip_r50_512x512_nwarner_open_images.py'
ckpt_file_maskclip = '/srv/essa-lab/flash3/nwarner30/MaskCLIP/pretrain/RN50_clip_backbone.pth' 

cfg_file_seg = '/coc/flash3/nwarner30/MaskCLIP/configs/maskclip_plus/zero_shot/maskclip_plus_r50_deeplabv2_r101-d8_512x512_80k_OpenImages.py'
ckpt_file_seg = '/srv/essa-lab/flash3/nwarner30/MaskCLIP/saved_runs/OpenImages 110 ZSS v3/iter_88000.pth'

maskclip_model, checkpoint = load_maskclip_model(cfg_file_maskclip, ckpt_file_maskclip)
seg_model = load_seg_model(cfg_file_seg, ckpt_file_seg, num_channels=5)


#ipdb.set_trace()
image_to_segmentation(INFERENCE_IMAGE_NAME,  click_points_list, OUT_NAME, maskclip_model, seg_model)
end_time = time.time()  # End timing
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Total execution time: {elapsed_time} seconds")   

