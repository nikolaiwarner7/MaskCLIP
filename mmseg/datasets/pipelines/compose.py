# Copyright (c) OpenMMLab. All rights reserved.
import collections

from mmcv.utils import build_from_cfg
from nwarner_common_utils import PRODUCING_MASKCLIP_DATA, EVALUATE_USING_CLIP
import numpy as np
from ..builder import PIPELINES

@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        # May need to be specific to produce-maskclip-maps script, with flag
        # For the index 7 bit
        # 2DO: add flag so this doesn't get called during actual training job

        if PRODUCING_MASKCLIP_DATA:
            if 'raw_gt_seg' not in self.transforms[8].keys:
                self.transforms[8].keys.append('raw_gt_seg')

        for i, t in enumerate(self.transforms):
            if isinstance(t, PIPELINES.get('Pad')) and EVALUATE_USING_CLIP:
                max_dim = np.max(data['img'].shape[:-1])
                t.size = (max_dim, max_dim)
            data = t(data)
            if data is None:
                return None

            # After loading annotation, get raw (unpadded) and store
            if i == 1 and PRODUCING_MASKCLIP_DATA:
                data['raw_gt_seg'] = data['gt_semantic_seg'].copy()
            elif i==1 and EVALUATE_USING_CLIP:
                data['raw_gt_seg'] = data['gt_semantic_seg'].copy()                
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
