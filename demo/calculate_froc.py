import os
import numpy as np
from maskrcnn_benchmark.data.datasets.evaluation.histo import calc_froc
import argparse


def cal_froc(model_name, iteration):
    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    # call load_data with allow_pickle implicitly set to true
    pr='/mnt/DATA_OTHER/digestPath/results/%s/inference/validation/recs_%07d.npy'%(model_name, iteration)
    pnr='/mnt/DATA_OTHER/digestPath/results/%s/inference/normal_region/nrfp_%07d.npy'%(model_name, iteration)
    nrfp = np.load(pnr)
    r= np.load(pr)
    # restore np.load for future normal usage
    np.load = np_load_old


    froc = calc_froc(r, nrfp)
    return froc
                
def run_cal_froc(model_id):
    models = []
    # ringcell
    for file in os.listdir('configs/ring_cell/'):
        if file.split('.')[-1]=='yaml':
            models.append(file.split('.')[0])

    result_str = ''

    for j, model in enumerate(models):
        if j==int(model_id):
            for iteration in range(2500,20001,2500):
                print('model (%d/%d)'%(j+1,len(models)) + ' iter%d'%iteration)
                result = cal_froc(model, iteration)
                result_str += "{:.4f}\t".format(
                    result[1]
                )

            with open('results/ringcell/' + models[j] + "_froc.txt", "w") as fid:
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
    run_cal_froc(args.model_id)
    