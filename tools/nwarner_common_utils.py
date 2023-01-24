from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics

CLASS_SPLITS = {'seen': list(range(1,15+1)),
                'unseen': list(range(16,20+1)),
                'all': list(range(1,20+1))}
NUM_VOC_CLASSES = 21 # Includes background
# Testing

# Parameters for producing Maskclip VOC maps loaded in mmseg/apis/single_gpu test test.py
PRODUCING_MASKCLIP_DATA = False  # Fixed flag for dataloader to store raw segs, only run when producing data

PRODUCE_MASKCLIP_MAPS_CONFIG = 'val' # 'train' or 'val'
SUBSAMPLE = 1400 # Number of samples to take for producing our maskclip-heatmap dataset
# 1464 in train, 

# Automatically handle split and subdirectories accordingly
if PRODUCE_MASKCLIP_MAPS_CONFIG == 'val':
    SPLIT = 'unseen'
elif PRODUCE_MASKCLIP_MAPS_CONFIG == 'train':
    SPLIT = 'seen'

OUT_ANNOTATION_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Annotations/'+PRODUCE_MASKCLIP_MAPS_CONFIG # 2DO: should specify folders for train or test
OUT_RGB_S_DIR = 'data/VOCdevkit/VOC2012/RGB_S_Images/' + PRODUCE_MASKCLIP_MAPS_CONFIG

def rgbs_pre_eval(self, preds, indices, class_num):
    """Collect eval result from each iteration.

    Args:
        preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
            after argmax, shape (N, H, W).
        indices (list[int] | int): the prediction related ground truth
            indices.

    Returns:
        list[torch.Tensor]: (area_intersect, area_union, area_prediction,
            area_ground_truth).
    """
    # In order to compat with batch inference
    if not isinstance(indices, list):
        indices = [indices]
    if not isinstance(preds, list):
        preds = [preds]

    pre_eval_results = []

    for pred, index in zip(preds, indices):
        seg_map = self.get_gt_seg_map_by_idx(index)
        # Replace the foreground with the class idx for IoU calc
        seg_map[seg_map==1] = class_num
        pred[pred==1] = class_num
        pre_eval_results.append(
            intersect_and_union(pred, seg_map, NUM_VOC_CLASSES,
                                self.ignore_index, self.label_map,
                                self.reduce_zero_label))

    return pre_eval_results