import os
import sys
import argparse
# set this file's path as first searching path
sys.path.insert(0, os.path.split(os.path.realpath(__file__))[0])

import torch
import numpy as np
import cv2

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES
from yolox.exp.build import get_exp_by_name
from yolox.utils import postprocess
# from ..utils.visualize import vis



COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)

class_names = COCO_CLASSES


class Detector():
    """ 图片检测器 """
    def __init__(self, model='yolox-s', ckpt='yolox_s.pth.tar'):
        super(Detector, self).__init__()

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.exp = get_exp_by_name(model)
        self.test_size = self.exp.test_size  # TODO: 改成图片自适应大小
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])



    def detect(self, raw_img, visual=True, conf=0.5):
        info = {}
        img, ratio = preproc(raw_img, self.test_size, COCO_MEAN, COCO_STD)
        info['raw_img'] = raw_img
        info['img'] = img

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre  # TODO:用户可更改
            )[0].cpu().numpy()

        info['boxes'] = outputs[:, 0:4]/ratio
        info['scores'] = outputs[:, 4] * outputs[:, 5]
        info['class_ids'] = outputs[:, 6]
        info['box_nums'] = outputs.shape[0]
        # 可视化绘图
        # if visual:
        #     info['visual'] = vis(info['raw_img'], info['boxes'], info['scores'], info['class_ids'], conf, COCO_CLASSES)
        return info

class PersonDetector(Detector):
    def __init__(self, model='yolox-s', ckpt='yolox_s.pth.tar'):
        Detector.__init__(self, model=model, ckpt=ckpt)

        self.filter_class = "person"
    
    def detect(self, raw_img, conf=0.5):
        info = {}
        img, ratio = preproc(raw_img, self.test_size, COCO_MEAN, COCO_STD)
        # info['raw_img'] = raw_img
        # info['img'] = img

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        xywh = []
        scores = []

        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre  # TODO:用户可更改
            )[0]
            if (outputs!=None):
                outputs = outputs.cpu().numpy()
                info['boxes'] = outputs[:, 0:4]/ratio
                info['scores'] = outputs[:, 4] * outputs[:, 5]
                info['class_ids'] = outputs[:, 6]
                # info['box_nums'] = outputs.shape[0]
                for xyxy, class_id, score  in zip(info['boxes'],info['class_ids'],info['scores']):
                    if class_names[int(class_id)] in self.filter_class and score > conf:
                        resonable_xywh = self.toRasonablexywh(xyxy)
                        xywh.append(resonable_xywh)
                        scores.append(score)
        return xywh, scores
            # return info
    
    def toRasonablexywh(self, xyxy):
        ### Some x1 y1 might be negatives
        x1, y1, x2, y2 = xyxy
        x1 = max(0,x1)
        y1 = max(0,y1)
        return [int(x1), int(y1), int(x2-x1), int(y2-y1)]
        


if __name__=='__main__':
    parser = argparse.ArgumentParser("YOLOX Demo")
    parser.add_argument(
        "--images_dir", default="", help="images_dir"
    )
    parser.add_argument(
        "--store_dir", default="", help="store_dir"
    )
    args=parser.parse_args()
    detector = PersonDetector(model='yolox-m', ckpt='/home/jing/workspace/my_mono_foll_ws/src/mono_tracking/scripts/yolox_descriptor/weights/yolox_m.pth.tar')

    image_dir = args.images_dir
    files = sorted(os.listdir(image_dir))
    for file in files:
        if not file.endswith(".jpg"):
            continue
        filename = os.path.join(image_dir, file)
        img = cv2.imread(filename)
        xywh,scores = detector.detect(img,conf=0.8)
        if(len(xywh) == 0):
            print("Bad:", filename)
            continue
        print(xywh, scores)
        def getEstimatedDistance(_xywh):
            return 917.8394165039062*0.54/float(_xywh[2])
        np.savetxt(os.path.join(args.store_dir, file.strip(".jpg")+"_eDis.txt"),[getEstimatedDistance(xywh[0])])
        # print("Good:", filename)
