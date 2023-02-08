# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import tempfile
import warnings

import shutil
import pdb
import mmcv
import numpy as np
import torch
from torch.utils.tensorboard._utils import figure_to_image
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from nwarner_common_utils import CLASS_SPLITS, SUBSAMPLE, OUT_ANNOTATION_DIR, OUT_RGB_S_DIR, SPLIT
from nwarner_common_utils import PRODUCING_MASKCLIP_DATA, VISUALIZING_TRAINED_MODEL
from nwarner_common_utils import CLIP_SIM_THRESHOLD_PRESENT, EVALUATE_AND_VISUALIZE
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

import matplotlib.pyplot as plt
import math


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model, # the RGB model
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={},
                    produce_maskclip_maps=False,
                    maskclip_clip_fair_eval=False,
                    rgbs_maskclip_model=None):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    if rgbs_maskclip_model:
        rgbs_maskclip_model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    # To do simple CLIP eval to detect present likely classes    
    CLIP_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    CLIP_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")    
    VOC_class_labels = list(dataset.CLASSES)
    VOC_class_labels.pop(0) # remove background
    # To debug multichannel model, use a subsample
    EVAL_DIR = 'test_imgs/'
    
    
    if EVALUATE_AND_VISUALIZE:
        shutil.rmtree(EVAL_DIR)
        os.mkdir(EVAL_DIR)   
    for batch_indices, data in zip(loader_indices, data_loader):
        batch_size = len(batch_indices)
        # Temporarily changed to fix bug in middle of dataloader
        #if batch_indices[0] < SUBSAMPLE and batch_indices[0] > 570:
        if batch_indices[0] < SUBSAMPLE:
            result = None
            with torch.no_grad():
                # Not sure why but it is loading data containers differntly depending on whether:
                # Produce data, train, evaluate
                # Depends if it is train or val set, too
                if VISUALIZING_TRAINED_MODEL:
                    # We do need grad for vis train
                    data['img'] = data['img'][0].cuda()
                    #data['gt_semantic_seg'] = data['gt_semantic_seg'].data
                    data['img_metas'] = data['img_metas'][0].data[0]
                elif not PRODUCING_MASKCLIP_DATA and not maskclip_clip_fair_eval:
                    data['img'] = data['img'][0]
                    data['gt_semantic_seg'] = data['gt_semantic_seg'][0]
                # Maintains default behavior, which rescales logits post inference
                # For our Maskclip experiments, we disable this behavior for double inference
                # IE -> 1) RGB-> Maskclip saliency
                if maskclip_clip_fair_eval:
                    data['img'] = data['img'][0]
                    data['img_metas'] = data['img_metas'][0].data[0]
                    img_metas = data['img_metas']
                    data['rescale'] = False
                    result = model(return_loss=False, **data)
                    data['rescale'] = False
                else:
                    result = model(return_loss=False, **data)

            if show or out_dir:
                # Allows access to train data in test format
                if VISUALIZING_TRAINED_MODEL:
                    img_tensor = data['img']
                    img_metas = data['img_metas']
                else:
                    img_tensor = data['img'].data[0]
                    if not maskclip_clip_fair_eval:
                        img_metas = data['img_metas'].data[0]
                if not VISUALIZING_TRAINED_MODEL:
                # We dont have 3D RGB images to convert to imgs 
                    if not PRODUCING_MASKCLIP_DATA:
                        img_tensor = torch.unsqueeze(img_tensor, 0)
                    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                    assert len(imgs) == len(img_metas)
                elif VISUALIZING_TRAINED_MODEL:
                    # Make the 3D img part for visualizing
                    img_tensor = img_tensor.permute(0,2,3,1)
                    imgs = img_tensor[:,:,:,:-1].detach().cpu().numpy()
                for img, img_meta in zip(imgs, img_metas):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if produce_maskclip_maps:
                        # 1) Find the GT classes from img_metas
                        #gt_annotations = data['gt_semantic_seg'].data[0]
                        gt_raw_annotation = data['raw_gt_seg'][0]
                        # Pre-padded from dataloader transforms step-> want raw
                        gt_classes = np.unique(gt_raw_annotation).tolist()
                        # Remove background, ignore (0,255)
                        if 0 in gt_classes:
                            gt_classes.remove(0)
                        if 255 in gt_classes:
                            gt_classes.remove(255)


                        # 2) Create a list of GT present classes
                        gt_present_classes = [cls for cls in gt_classes if cls in CLASS_SPLITS[SPLIT]]

                        cls_specific_gt_seg = None
                        img_with_cls_logit = None
                        cls_logit = None
                        norm_cls_logit = None
                        # 3) Loop through said list
                        for cls in gt_present_classes:
                            if out_dir:
                                # 4) Grab the corresponding class-specific segmentation
                                cls_specific_gt_seg = gt_raw_annotation == cls
                                cls_specific_gt_seg = torch.tensor(cls_specific_gt_seg.numpy().astype(int))

                                out_name = img_meta['ori_filename']
                                out_name_with_cls = out_name.replace('.jpg','_class%s.npy' % cls)
                                np.save(OUT_ANNOTATION_DIR+'/'+out_name_with_cls, cls_specific_gt_seg)    

                                # 0 indexing for class (cls)
                                cls_logit = result[0][1][0, cls-1].detach().cpu()
                                cls_logit = np.array(cls_logit)


                                ## Normalize class logits:
                                # From 0-1 for preproc later
                                norm_cls_logit = (cls_logit - np.min(cls_logit)) / (np.max(cls_logit)-np.min(cls_logit))

                                # Combine with full size image
                                img_with_cls_logit = np.concatenate((img_show, np.expand_dims(norm_cls_logit, -1)), axis=-1)


                                # Write this out using np.save
                                np.save(OUT_RGB_S_DIR+'/'+out_name_with_cls, img_with_cls_logit)

                                out_file = osp.join(out_dir, out_name_with_cls)
                                
                            else:
                                out_file = None

                                model.module.show_result(
                                    img_show,
                                    result,
                                    palette=dataset.PALETTE,
                                    show=show,
                                    out_file=out_file,
                                    opacity=opacity,
                                    produce_maskclip_maps_class_id=cls,
                                    )

                        del img_tensor, img_metas, imgs, img_show
                        del gt_raw_annotation, gt_classes, gt_present_classes, cls_specific_gt_seg
                        del cls_logit, norm_cls_logit, img_with_cls_logit

                    elif maskclip_clip_fair_eval:
                        # Get the predicted maskclip logits
                        # It is BS=1, NUM_CLASSES=20, SP_DIM, SP_DIM
                        maskclip_logits = result[0][1]

                        original_rgb_img = data['img'].data[0]

                        h, w, _ = img_meta['img_shape']
                        img_show = img[:h, :w, :]

                        ori_h, ori_w = img_meta['ori_shape'][:-1]
                        #Img show is resized appropriately without padding
                        img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                        CLIP_input_img = img_show
                        #CLIP_input_img = original_rgb_img.permute(0, 2, 3, 1)[0]


                        # Finding GT present class list using prompts:
                        CLIP_inputs = CLIP_processor(text= VOC_class_labels, images=CLIP_input_img, return_tensors="pt", padding=True)
                        CLIP_outputs = CLIP_model(**CLIP_inputs)
                        logits_per_image = CLIP_outputs.logits_per_image
                        potential_present_idxs = np.where(logits_per_image>CLIP_SIM_THRESHOLD_PRESENT)[1].tolist()
                        if not potential_present_idxs:
                            potential_present_idxs = [np.argmax(logits_per_image.detach().cpu().numpy())]
                        # For human reference: 
                        potential_classes = [VOC_class_labels[i] for i in potential_present_idxs]

                        # 3) Loop through said list8
                        # Force seg list for debugging
                        predicted_segs = []
                        #pdb.set_trace()
                        for cls in potential_present_idxs:
                            # Grab the corresponding maskclip heatmap
                            cls_heatmap = maskclip_logits[0,cls].detach().cpu().numpy()
                            # Normalize between -1, 1 to match training
                            cls_normalized_heatmap = (cls_heatmap-(np.max(cls_heatmap)+np.min(cls_heatmap))/2)/\
                                ((np.max(cls_heatmap)-np.min(cls_heatmap))/2)
                            padded_normalized_hmap = mmcv.impad(cls_normalized_heatmap,\
                                 shape=(512,512), pad_val=0)
                            # Ideally heatmaps are between 0-1 
                            #padded_normalized_hmap = mmcv.image.imnormalize_(padded_hmap\
                            #    , mean=np.array(0.5), std=np.array(0.5), to_rgb=False)
                            # Just min max normalize

                            # Combine with full size image
                            if len(original_rgb_img.shape)<4:
                                original_rgb_img = torch.unsqueeze(original_rgb_img,0)
                            img_with_cls_logit = np.concatenate((original_rgb_img, np.expand_dims(padded_normalized_hmap, (0,1))), axis=1)
                            #Try, now substitute this for the data batch in the rgbs model
                            data['img'] = torch.tensor(img_with_cls_logit).cuda()
                            #data['gt_semantic_seg'] = cls_specific_gt_seg
                            result = rgbs_maskclip_model(return_loss=False, **data)

                            predicted_cls_seg = result[0][0][0]
                            mask_area = np.sum(predicted_cls_seg)
                            predicted_segs.append((mask_area, cls, predicted_cls_seg))

                        # Sort predicted segs by decreasing area, append final result
                        predicted_segs.sort(reverse = True)
                        multiclass_seg = np.zeros(predicted_cls_seg.shape)
                        for (area, cls, fgbg_mask) in predicted_segs:
                            multiclass_seg[fgbg_mask==1] = cls+1 #previous were 0 index to maintain bg class

                        gt_present_classes = np.unique(data['gt_semantic_seg'][0].data[0].detach().cpu().numpy()).tolist()

                        # Make sure it only contains list of foreground classes                    
                        if 0 in gt_present_classes:
                            gt_present_classes.remove(0)
                        if 255 in gt_present_classes:
                            gt_present_classes.remove(255)

                        # Check if there are >1 gt classes in an image, visualize those images
                        #if EVALUATE_AND_VISUALIZE and len(gt_present_classes)>1:

                        # Check if there are unseen classes in gt present in image, visualize these:
                        if EVALUATE_AND_VISUALIZE and set(gt_present_classes).intersection(CLASS_SPLITS['unseen']):
                            # Input saliency 
                            test_img = padded_normalized_hmap
                            test_img = (test_img -np.min(test_img))/ (np.max(test_img)-np.min(test_img))
                            filename = img_metas[0]['ori_filename'].split(".jpg")[0]
                            plt.imsave('test_imgs/%s_saliency_input.png' % filename, test_img)

                            # Input rgb
                            img = original_rgb_img
                            test = img.permute(0, 2, 3, 1)
                            test_img = test[0,:,:,:].detach().cpu().numpy()
                            # Pick img_num for first ch
                            test_img = (test_img -np.min(test_img))/ (np.max(test_img)-np.min(test_img))
                            filename = img_metas[0]['ori_filename'].split(".jpg")[0]
                            final_classes = '_'.join(potential_classes)
                            plt.imsave('test_imgs/%s_%s_rgb_input.png' % (filename, final_classes), test_img)

                            # Output pred seg
                            test_img = multiclass_seg
                            test_img = (test_img -np.min(test_img))/ (np.max(test_img)-np.min(test_img))
                            filename = img_metas[0]['ori_filename'].split(".jpg")[0]
                            plt.imsave('test_imgs/%s_pred_seg.png' % filename, test_img)

                            # Output GT Seg
                            test_img = data['gt_semantic_seg'][0][0,0]
                            test_img[test_img==255]=0
                            filename = img_metas[0]['ori_filename'].split(".jpg")[0]
                            plt.imsave('test_imgs/%s_gtseg.png' % filename, test_img)

                                                
                    if VISUALIZING_TRAINED_MODEL:
                        # Get filename with class
                        filename = data['img_metas'][0]['ori_filename']
                        filename = filename.split(".npy")[0]

                        # Save the original unscaled image
                        test_img = img_show
                        test_img = (test_img -np.min(test_img))/ (np.max(test_img)-np.min(test_img))
                        plt.imsave(out_dir+filename+'_input_img.png', test_img)

                        # Save the seg prediction
                        test_out = result[0][0][0]
                        plt.imsave(out_dir+filename+'_pred_seg.png', test_out)

                        # Save the GT segmentation
                        gt_seg = data['gt_semantic_seg'][0]
                        gt_seg = gt_seg.detach().cpu().numpy()
                        plt.imsave(out_dir+filename+'_gt_seg.png', gt_seg[0,0,:,:]==1)

                if efficient_test:
                    result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]



            if pre_eval and maskclip_clip_fair_eval:
                # We made inference for val at 512,512 res w padding (once)
                # resize multiclass_seg prediction to actual input img res for fair eval
                # Unresize to max original dimension, then unpad
                or_h = img_metas[0]['ori_shape'][0]
                or_w = img_metas[0]['ori_shape'][1]
                max_dim = np.max([or_h, or_w])
                multiclass_seg = mmcv.imrescale(multiclass_seg, (max_dim, max_dim))
                # cut out padding by cropping unscaled to original dims
                unpadded_multiclass_seg = multiclass_seg[:or_h, :or_w]
                result = dataset.pre_eval(unpadded_multiclass_seg, indices=batch_indices)
                results.extend(result)
            else :
                pass



            if format_only:
                result = dataset.format_results(
                    result, indices=batch_indices, **format_args)
            # pre eval shouldnt run while producing data

            if not PRODUCING_MASKCLIP_DATA and not maskclip_clip_fair_eval:
                if pre_eval:
                    # TODO: adapt samples_per_gpu > 1.
                    # only samples_per_gpu=1 valid now
                    # This is to correct from training data loader mixed in with test

                    # Look at filename:
                    if VISUALIZING_TRAINED_MODEL:
                        filename = data['img_metas'][0]['ori_filename']
                    elif maskclip_clip_fair_eval:
                        filename = data['img_metas'].data[0][0]['filename']
                    else:
                        filename = data['img_metas'][0].data[0][0]['ori_filename']
                    target_label = filename.split("class")
                    # Grab the target class num  (1-20, not 0 idxs)
                    target_label = int(target_label[1].strip(".npy"))


                    seg_pred = result[0][0][0]
                    result = dataset.rgbs_pre_eval(seg_pred, indices=batch_indices, class_num=target_label)
                    results.extend(result)
                else :
                    results.extend(result)

                batch_size = len(result)
            # Avoids memory leak from extracting raw logits
            del result

        for _ in range(batch_size):
            prog_bar.update()
    if not PRODUCING_MASKCLIP_DATA:
        dataset.evaluate(results)
        return results
    else:
        return None


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={}):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            data['img'] = data['img'][0]
            data['gt_semantic_seg'] = data['gt_semantic_seg'][0]
            result = model(return_loss=False, **data)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            filename = data['img_metas'][0].data[0][0]['ori_filename']
            target_label = filename.split("class")
            target_label = int(target_label[1].strip(".npy"))
            seg_pred = result[0][0][0]

            result = dataset.rgbs_pre_eval(seg_pred, indices=batch_indices, class_num=target_label)
            results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def vis_output(model, data_loader, config_name, num_vis, 
                    highlight_rule, black_bg, pavi=False):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    if pavi:
        from pavi import SummaryWriter
        writer = SummaryWriter(config_name, project='maskclip')
    else:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('vis/{}'.format(config_name))

    if len(highlight_rule) == 0:
        highlight_names = []
    elif highlight_rule == 'zs5':
        highlight_names = [10, 14, 1, 18, 8, 20, 19, 5, 9, 16]
    else:
        rank, splits, strategy = highlight_rule.split('_')
        rank, splits = int(rank), int(splits)
        all_index = list(range(len(class_names)))
        if strategy == 'itv':
            highlight_names = all_index[(rank-1)::splits]
        elif strategy == 'ctn':
            classes_per_split = len(class_names) // splits
            highlight_names = all_index[(rank-1)*classes_per_split : rank*classes_per_split]

    count = 0
    class_names = list(dataset.CLASSES)
    palette = dataset.PALETTE
    for batch_indices, data in zip(loader_indices, data_loader):
        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0].data[0]
        gt = dataset.get_gt_seg_map_by_idx(batch_indices[0]) if black_bg else None

        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        assert len(imgs) == len(img_metas)

        img, img_meta = imgs[0], img_metas[0]

        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]
        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        with torch.no_grad():
            seg_logit = model(return_loss=False, rescale=True, return_logit=True, **data)
        seg_logit = seg_logit[0]
        seg_pred = seg_logit.argmax(axis=0)

        if len(class_names) + 1 == seg_logit.shape[0] and count == 0:
            class_names = ['background'] + class_names
            palette = [(0, 0, 0)] + palette
            highlight_names = [i+1 for i in highlight_names]

        img_seg = model.module.show_result(img_show, [seg_pred], opacity=0.5,
                                            palette=palette,
                                            classes=class_names, gt=gt)

        filename = img_metas[0]['ori_filename']
        # seg_logit = np.exp(seg_logit*100)
        seg_logit = (seg_logit == seg_logit.max(axis=0, keepdims=True))
        fig = activation_matplotlib(seg_logit, img_show, img_seg, class_names, highlight_names)
        # writer.add_figure(filename, fig)
        img = figure_to_image(fig)
        writer.add_image(filename, img)

        batch_size = img_tensor.size(0)
        for _ in range(batch_size):
            prog_bar.update()

        count += 1
        if count == num_vis:
            break
    writer.close()


def activation_matplotlib(seg_logit, image, image_seg, class_names, highlight_names):
    total = len(class_names)+1 if image_seg is None else len(class_names)+2
    row, col = math.ceil((total)/5), 5
    fig = plt.figure(figsize=(6*col, 3*row))
    count, class_idx = 0, 0
    for _ in range(row):
        for _ in range(col):
            if count == 0:
                ax = fig.add_subplot(row, col, count+1, xticks=[], yticks=[], title='image')
                plt.imshow(image[..., ::-1])
                count += 1
                if image_seg is not None:
                    ax = fig.add_subplot(row, col, count+1, xticks=[], yticks=[], title='seg')
                    plt.imshow(image_seg[..., ::-1])
                    count += 1
            
            if count == total:
                return fig

            ax = fig.add_subplot(row, col, count+1, xticks=[], yticks=[], title=class_names[class_idx])
            if class_idx in highlight_names:
                ax.set_title(class_names[class_idx], color='r')
            plt.imshow(seg_logit[class_idx])
            count += 1
            class_idx += 1
    
    # fig.tight_layout()
    return fig