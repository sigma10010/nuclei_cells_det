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


def test_net_on_wsi(config_file, ckpt, iteration):
    #========test on monuseg
    img_root = '/mnt/DATA_OTHER/moNuSeg/original_testing/tissue_Images/'
    anno_root = '/mnt/DATA_OTHER/moNuSeg/original_testing/tissue_Images/'
#     img_root = '/mnt/DATA_OTHER/moNuSeg/original_training/tissue_Images/'
#     anno_root = '/mnt/DATA_OTHER/moNuSeg/original_training/tissue_Images/'
    test_data = monuseg(img_root, anno_root)
    
#     ###========test on monuseg
#     img_root = '/mnt/DATA_OTHER/digestPath/Signet_ring_cell_dataset/sig-train-pos/validation/'
#     test_data = ringcell(img_root, anno_root)

    _, img_list, _, _ = test_data.walk_root_dir()

#     config_file = args.config_file
    # config_file = '/home/ys309/Dropbox/coding/maskrcnn-benchmark/configs/monuseg/rc_retinanet_R-50-FPN_1x.yaml'
    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda:0"])
    cfg.MODEL.WEIGHT = ckpt
    # cfg.MODEL.WEIGHT = '/mnt/DATA_OTHER/moNuSeg/results/rc_retinanet_R-50-FPN_1x/model_0020000.pth'

    histo_demo = HistoDemo(
        cfg,
        min_image_size=600,
        confidence_threshold=0.5,
    )

    # ### confidence_threshold=0.5

    # # plot bounding boxes
    # from maskrcnn_benchmark.data.datasets.evaluation.histo import eval_detection_monuseg
    # model_size = 600
    # overlap = 200
    # # l_scores=[]
    # for i, fimg in enumerate(img_list):
    #     if i<90:
    #         print('%d/%d'%(i+1,len(img_list)))
    #         img = Image.open(fimg)
    #         boxlist=histo_demo.sliding_window_wsi(img, nms_thresh = 0.25)
    #     #     l_scores.append(boxlist.get_field('scores'))
    #         boxes=boxlist.bbox.cpu().numpy()

    #         gt = test_data.get_groundtruth(i)
    #         gt_boxes = gt.bbox.cpu().numpy()

    #         result = eval_detection_monuseg([boxlist], [gt], iou_thresh=0.25,)
    #         print (result)
    #     else:
    #         break
    # # 
    # # R-50-2500

    ### confidence_threshold=0.5
    # all testing data

    # plot bounding boxes
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
    output_folder = os.path.join(cfg.OUTPUT_DIR, cfg.OUTPUT_FOLDER)
    result = eval_detection_monuseg(predictions, gts, iou_thresh=0.3, iteration=iteration, output_folder=output_folder)
    print (result)
    del histo_demo
    return result
            
def run_test_net(model_id):
    cfgs = []
    models = []
#     # ringcell
#     for file in os.listdir('configs/ring_cell/'):
#         cfgs.append(file)
#         models.append(file.split('.')[0])
        
    # monuseg
    for file in os.listdir('configs/monuseg/'):
        if file.split('_')[1] == 'retinanet':
            cfgs.append(file)
            models.append(file.split('.')[0])

    result_str = ''

    for j, cfg in enumerate(cfgs):
        if j==int(model_id):
#             aps = []
#             f1s = []
            for iteration in range(2500,20001,2500):
                print('model (%d/%d)'%(j+1,len(models)) + ' iter%d'%iteration)
                config_file = '/home/ys309/Dropbox/coding/maskrcnn-benchmark/configs/monuseg/%s'%cfgs[j]
                ckpt = '/mnt/DATA_OTHER/moNuSeg/results/'+models[j]+'/model_%07d.pth'%iteration
                print(config_file + '\n' + ckpt)

                result = test_net_on_wsi(config_file, ckpt, iteration)
#                 aps.append(result["ap"][1])
#                 f1s.append(result["f1"][1])


#                 result_str = models[i] + ':\n'
#                 result_str += "ap:"
                for i, ap in enumerate(result["ap"]):
                    if i == 0:  # skip background
                        continue
                    result_str += "{:.4f}\t".format(
                        ap
                    )
#                 result_str += "f1:"
                for i, f1 in enumerate(result["f1"]):
                    if i == 0:  # skip background
                        continue
                    result_str += "{:.4f}\t".format(
                        f1
                    )
#                 result_str += "TP FP FN:"
                for i, confusion in enumerate(result["confusion"]):
                    if i == 0:  # skip background
                        continue
                    result_str += "{:d}\t {:d}\t {:d}\n".format(
                      confusion[0], confusion[1], confusion[2]
                    )
                print(result_str)
                
            with open('results/monuseg/' + models[j] + ".txt", "w") as fid:
                fid.write(result_str)
#             results_dict_aps.update({models[i]: aps})
#             results_dict_f1s.update({models[i]: f1s})
        else:
            continue
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--model_id",
        default=None,
        help="[0,8]",
    )
#     parser.add_argument(
#         "--ckpt",
#         help="The path to the checkpoint for test, default is the latest checkpoint.",
#         default=None,
#     )
    args = parser.parse_args()
    
#     test_net_on_wsi(args.config_file, args.ckpt)
    run_test_net(args.model_id)