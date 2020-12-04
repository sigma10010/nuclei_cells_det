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
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from predictor import HistoDemo
import argparse

def test_net_on_nr(config_file, ckpt, iteration):
    path_image = '/mnt/DATA_OTHER/digestPath/Signet_ring_cell_dataset/sig-train-neg/original/'
    img_list = os.listdir(path_image)
    # config_file = '/home/ys309/Dropbox/coding/maskrcnn-benchmark/configs/ring_cell/rc_faster_rcnn_R_50_FPN_1x_rpn_pair.yaml'
    # update the config options with the config file
    cfg.merge_from_file(config_file)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda:0"])
    # cfg.MODEL.WEIGHT = '/mnt/DATA_OTHER/digestPath/results/rc_faster_rcnn_R_50_FPN_1x_rpn_pair/model_0005000.pth'
    cfg.MODEL.WEIGHT = ckpt
    output_dir=cfg.OUTPUT_DIR
    # print(output_dir)
    output_fold=cfg.OUTPUT_FOLDER
    save_fold=os.path.join(output_dir, output_fold, 'inference', 'normal_region')
    mkdir(save_fold)

    histo_demo = HistoDemo(
        cfg,
        min_image_size=600,
        confidence_threshold=0.5,
    )

    model_size = 600
    overlap = 200
    l_scores=[]
    for i, fimg in enumerate(img_list):
        print('%d/%d'%(i+1,len(img_list)))
        fimg = Image.open(path_image+'/'+img_list[i])
        boxlist=histo_demo.sliding_window_wsi(fimg)
        if boxlist is not None:
            print('there are some FP')
            l_scores.append(boxlist.get_field('scores'))
        else:
            print('boxlist is None')
    if l_scores:
        scores = torch.cat(l_scores,0) 
        np.save(os.path.join(save_fold, 'scores_%07d.npy'%iteration), scores.cpu().numpy())

        score_l = scores.cpu().numpy()
        
        n_imgs = len(img_list)
        nrfp_conf_thres = (score_l>histo_demo.confidence_threshold).sum()//n_imgs
        fps_conf_thres = np.maximum(100 - nrfp_conf_thres, 0)
        # save fps for froc calculation
        nrfp = []
        fps = []
        # range same as the one for recs
        for conf_t in np.arange(0.0, 1.0, 0.001):
            nrfp.append((score_l>conf_t).sum()//n_imgs)
            fps.append(np.maximum(100 - (score_l>conf_t).sum()//n_imgs, 0))
        np.save(os.path.join(save_fold, 'nrfp_%07d.npy'%iteration), np.array(nrfp))
        np.save(os.path.join(save_fold, 'fps_%07d.npy'%iteration), np.array(fps))

    return { "nrfp": nrfp_conf_thres,  "fps": fps_conf_thres}
                
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

                result = test_net_on_nr(config_file, ckpt, iteration)
                result_str += "{:.4f}\t".format(
                    result["fps"]
                )

            with open('results/ringcell/' + models[j] + "_nr.txt", "w") as fid:
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
    