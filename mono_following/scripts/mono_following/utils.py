from __future__ import absolute_import
import torch

GLOBAL = 'globl'
FOREGROUND = 'foreg'
BACKGROUND = 'backg'
CONCAT_PARTS = 'conct'
PARTS = 'parts'
BN_GLOBAL = 'bn_globl'
BN_FOREGROUND = 'bn_foreg'
BN_BACKGROUND = 'bn_backg'
BN_CONCAT_PARTS = 'bn_conct'
BN_PARTS = 'bn_parts'
PIXELS = 'pixls'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def maybe_cuda(what, use_cuda=True, **kw):
    """
    Moves `what` to CUDA and returns it, if `use_cuda` and it's available.
        Args:
            what (object): any object to move to eventually gpu
            use_cuda (bool): if we want to use gpu or cpu.
        Returns
            object: the same object but eventually moved to gpu.
    """

    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda()
    return what

def mini_batch_deep_part_features(model, total_x, num, total_vis_map, is_vis_att_map=False, batch_size=512):
    """
        Compute deep features with mini-batches.
            Args:
                model (object): neural network.
                total_x (tensor): data tensor.
                num (int): number of data.
                vis_map (tensor): vis part map with shape (N,4,W,H)
            Returns
                deep_features (tensor): deep feature representation of data tensor.
    """
    is_train = False
    if model.training:
        is_train = True
        model.eval()

    with torch.no_grad():
        bs = batch_size
        num_itr = num // bs + int(num % bs > 0)
        sid = 0
        deep_features_list = []
        att_map_list = []
        for i in range(num_itr):
            eid = sid + bs if i != num_itr - 1 else num
            batch_x = total_x[sid: eid]
            vis_map = total_vis_map[sid: eid]
            
            if is_vis_att_map:
                batch_deep_features_, att_map = model.extract_features(batch_x, vis_map, True)
                att_map_list.append(att_map)
            else:
                batch_deep_features_ = model.extract_features(batch_x, vis_map, False)

            deep_features_list.append(batch_deep_features_)
            sid = eid
        if is_vis_att_map:
            att_maps_ = att_map_list[0] if num_itr == 1 else torch.cat(att_map_list, 0)
        deep_features_ = deep_features_list[0] if num_itr == 1 else torch.cat(deep_features_list, 0)
    if is_train:
        model.train()
    if is_vis_att_map:
        return deep_features_, att_maps_
    return deep_features_