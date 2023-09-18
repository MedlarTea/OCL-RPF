import torch
import numpy as np
def get_iou(self, pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
        (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
        inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou

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


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


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

class MyAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = []

    def get_u_std(self, window_size):
        mean = np.mean(self.data[-window_size:])
        # mse = np.mean((self.data[-window_size:] - mean)**2)
        # rms = np.sqrt(mse)
        std = np.var(self.data[-window_size:])
        return mean, max(std, 1e-6)

    def update(self, val, n=1):
        self.val = val
        self.data.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mini_batch_deep_features(model, total_x, num, batch_size=512):
    """
        Compute deep features with mini-batches.
            Args:
                model (object): neural network.
                total_x (tensor): data tensor.
                num (int): number of data.
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
        for i in range(num_itr):
            eid = sid + bs if i != num_itr - 1 else num
            batch_x = total_x[sid: eid]

            # if model_has_feature_extractor:
            #     batch_deep_features_ = model.features(batch_x)
            # else:
            #     batch_deep_features_ = torch.squeeze(model_features(batch_x))
            batch_deep_features_ = model.extract_features(batch_x)

            deep_features_list.append(batch_deep_features_.reshape((batch_x.size(0), -1)))
            sid = eid
        if num_itr == 1:
            deep_features_ = deep_features_list[0]
        else:
            deep_features_ = torch.cat(deep_features_list, 0)
    if is_train:
        model.train()
    return deep_features_

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


def euclidean_distance(u, v):
    euclidean_distance_ = (u - v).pow(2).sum(1)
    return euclidean_distance_


def ohe_label(label_tensor, dim, device="cpu"):
    # Returns one-hot-encoding of input label tensor
    n_labels = label_tensor.size(0)
    zero_tensor = torch.zeros((n_labels, dim), device=device, dtype=torch.long)
    return zero_tensor.scatter_(1, label_tensor.reshape((n_labels, 1)), 1)


def nonzero_indices(bool_mask_tensor):
    # Returns tensor which contains indices of nonzero elements in bool_mask_tensor
    return bool_mask_tensor.nonzero(as_tuple=True)[0]


class EarlyStopping():
    def __init__(self, min_delta, patience, cumulative_delta):
        self.min_delta = min_delta
        self.patience = patience
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score = None

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False

    def reset(self):
        self.counter = 0
        self.best_score = None

