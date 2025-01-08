import matplotlib.pyplot as plt
import numpy as np
from torch import sort
import os
import cv2
import math

### THIRD VERSION ###
datasets = ["dataset3"]
_dir = "/home/jing/Data/Dataset/FOLLOWING/evaluation-of-width-distance/motionCapture"
for dataset in datasets:
    files_dir = os.path.join(_dir, dataset)
    files = sorted(os.listdir(files_dir))

    gts = []
    ess = []
    kfs = []
    errors_es = []
    errors_kf = []
    for file in files:
        if not file.endswith(".txt"):
            continue
        filename = os.path.join(files_dir, file)
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        gt = fs.getNode("gt").mat()[0,0]
        es = math.sqrt(pow(fs.getNode("es").mat()[0,0],2)+pow(fs.getNode("es").mat()[0,1],2))
        kf = math.sqrt(pow(fs.getNode("kf").mat()[0,0],2)+pow(fs.getNode("kf").mat()[0,1],2))
        gts.append(gt)
        ess.append(es)
        kfs.append(kf)
        errors_es.append(abs(gt-es+0.08))
        errors_kf.append(abs(gt-kf+0.08))

    plt.rcParams['font.family'] = ['Times New Roman']
    plt.figure(figsize=(3.45,2))
    plt.plot(gts, errors_es, color="orange", label="w/o KF")
    plt.plot(gts, errors_kf, color="royalblue", label="w/ KF")
    plt.ylabel('Error (m)', fontsize=8)
    plt.xlabel('Distance (m)', fontsize=8)
    # plt.scatter(gts, errors_es, color="red", label="es")
    # plt.scatter(gts, errors_kf, color="blue", label="kf")
    plt.yticks(np.arange(0.0,2.5,0.5), fontsize=6)
    plt.xticks(np.arange(0.0,7.0,1.0), fontsize=6)
    plt.gcf().subplots_adjust(left=0.14)
    plt.gcf().subplots_adjust(bottom=0.16)
    plt.gcf().subplots_adjust(right=0.99)
    plt.gcf().subplots_adjust(top=0.95)
    plt.legend(prop={'size':6})
    plt.savefig(os.path.join(_dir, dataset+".pdf"))
    # plt.show()