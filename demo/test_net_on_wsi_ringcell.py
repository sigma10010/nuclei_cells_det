import os
import numpy as np
import pylab as plt
from PIL import Image,ImageDraw
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import requests
from io import BytesIO
import matplotlib.pyplot as pyplt
import torch
import argparse
from maskrcnn_benchmark.data.datasets.evaluation.histo import eval_detection_monuseg, eval_detection_ringcell

from maskrcnn_benchmark.data.datasets import MoNuSegDetection as monuseg
from maskrcnn_benchmark.data.datasets import RingCellDetection as ringcell

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from predictor import HistoDemo

from maskrcnn_benchmark.utils.miscellaneous import mkdir


def test_net_on_wsi(config_file, ckpt, iteration):
    ###========test on monuseg
    img_root = '/mnt/DATA_OTHER/digestPath/Signet_ring_cell_dataset/sig-train-pos/validation/'
    test_data = ringcell(img_root)

    _, img_list, _, _ = test_data.walk_root_dir()

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda:0"])
    cfg.MODEL.WEIGHT = ckpt
    
    output_dir=cfg.OUTPUT_DIR
    # print(output_dir)
    output_fold=cfg.OUTPUT_FOLDER
    save_fold=os.path.join(output_dir, output_fold, 'inference', 'validation')
    mkdir(save_fold)

    histo_demo = HistoDemo(
        cfg,
        min_image_size=600,
        confidence_threshold=0.5,
    )

    model_size = 600
    overlap = 200
    # l_scores=[]
    predictions = []
    gts = []

    for i, fimg in enumerate(img_list):
        if i<90:
            print('%d/%d'%(i+1,len(img_list)))
            img = Image.open(fimg)
            boxlist=histo_demo.sliding_window_wsi(img, nms_thresh = 0.3)
            predictions.append(boxlist)
            gt = test_data.get_groundtruth(i)
            gts.append(gt)   
        else:
            break

    result = eval_detection_ringcell(predictions, gts, iou_thresh=0.3)
    # save rec for froc calculation
    np.save(os.path.join(save_fold, "recs_%07d.npy"%iteration),np.array(result["recs"]))
    
    print (result)
    del histo_demo
    return result
            
def run_test_net(model_id):
    cfgs = []
    models = []
    # ringcell
    for file in os.listdir('configs/ring_cell/'):
        if file.split('.')[-1]=='yaml':
            cfgs.append(file)
            models.append(file.split('.')[0])

    result_str = ''

    for j, cfg in enumerate(cfgs):
        if j==int(model_id):
            for iteration in range(2500,20001,2500):
                print('model (%d/%d)'%(j+1,len(models)) + ' iter%d'%iteration)
                config_file = '/home/ys309/Dropbox/coding/maskrcnn-benchmark/configs/ring_cell/%s'%cfgs[j]
                ckpt = '/mnt/DATA_OTHER/digestPath/results/'+models[j]+'/model_%07d.pth'%iteration
                print(config_file + '\n' + ckpt)

                result = test_net_on_wsi(config_file, ckpt, iteration)

                for i, rec_conf_thres in enumerate(result["rec_conf_thres"]):
                    if i == 0:  # skip background
                        continue
                    result_str += "{:.4f}\t".format(
                        rec_conf_thres
                    )

            with open('results/ringcell/' + models[j] + ".txt", "w") as fid:
                fid.write(result_str)
        else:
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--model_id",
        default=None,
        help="[0,17]",
    )
#     parser.add_argument(
#         "--ckpt",
#         help="The path to the checkpoint for test, default is the latest checkpoint.",
#         default=None,
#     )
    args = parser.parse_args()
    
#     test_net_on_wsi(args.config_file, args.ckpt)
    run_test_net(args.model_id)