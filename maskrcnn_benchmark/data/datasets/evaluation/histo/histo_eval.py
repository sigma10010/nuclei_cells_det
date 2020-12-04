# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

def do_monuseg_evaluation(dataset, predictions, output_folder, logger):
    #====== iou_thresh=0.25, confidence_threshold = 0.5 ======
    # TODO need to make the use_07_metric format available
    # for the user to choose
    pred_boxlists = []
    gt_boxlists = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)
    result = eval_detection_monuseg(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.25,
        use_07_metric=True,
        output_folder=output_folder
    )
    result_str = "mAP: {:.4f}\n".format(result["map"])
    
    result_str += "ap:\n"
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i), ap
        )
        
    result_str += "f1:\n"
    for i, f1 in enumerate(result["f1"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i), f1
        )
    logger.info(result_str)
    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str)
    return result


def eval_detection_monuseg(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False, iteration=0, output_folder=''):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    
    prec, rec, f1, confusion = calc_detection_monuseg_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    # save prec and rec for precision-recall curve
    np.save(os.path.join(output_folder, "prec_%07d.npy"%iteration),prec[1])
    np.save(os.path.join(output_folder, "rec_%07d.npy"%iteration),rec[1])
    
    ap = calc_detection_monuseg_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap), "f1": f1, "confusion": confusion}


def calc_detection_monuseg_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5, confidence_threshold = 0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        if pred_boxlist.has_field('objectness'):
            pred_score = pred_boxlist.get_field("objectness").numpy()
        else:
            pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    
    #!!!!!!!!!!!!!!!!!
    # added by ys
    rec_conf_thres = [None] * n_fg_class
    prec_conf_thres = [None] * n_fg_class
    f1_conf_thres = [None] * n_fg_class
    confusion_conf_thres = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]
        
        # calculate tp/fp at fix confidence_threshold
        not_match_mask_l = match_l == 0
        not_match_score_l = score_l[not_match_mask_l]
        fp_conf_thres = (not_match_score_l>confidence_threshold).sum()
        
        match_mask_l = match_l == 1
        match_score_l = score_l[match_mask_l]
        tp_conf_thres = (match_score_l>confidence_threshold).sum()
        
        prec_conf_thres[l] = tp_conf_thres/(tp_conf_thres + fp_conf_thres)
        
        if n_pos[l] > 0:
            rec_conf_thres[l] = tp_conf_thres / n_pos[l]
            tn_conf_thres = n_pos[l] - tp_conf_thres
            
        # F1
        f1_conf_thres[l] = 2*prec_conf_thres[l]*rec_conf_thres[l]/(prec_conf_thres[l] + rec_conf_thres[l])
        confusion_conf_thres[l] = [tp_conf_thres, fp_conf_thres, tn_conf_thres]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec, f1_conf_thres, confusion_conf_thres


def calc_detection_monuseg_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

#===============================================================

def do_ringcell_evaluation(dataset, predictions, output_folder, logger, dataset_nr = None, predictions_nr = None):
    # TODO need to make the use_07_metric format available
    # for the user to choose
    pred_boxlists_nr = []
    pred_boxlists = []
    gt_boxlists = []
    if dataset_nr is not None and predictions_nr is not None:
        for image_id, prediction_nr in enumerate(predictions_nr):
            img_info = dataset_nr.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            prediction_nr = prediction_nr.resize((image_width, image_height))
            pred_boxlists_nr.append(prediction_nr)
    else:
        pred_boxlists_nr = None
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)
        
    result = eval_detection_ringcell(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.3,
        use_07_metric=True,
        pred_boxlists_nr=pred_boxlists_nr,
    )
    
    result_str = "mAP: {:.4f}\n".format(result["map"])
    result_str += "ap:\n"
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i), ap
        )
    # !!!!!!!!!!
    # calculate froc
    path2nr = os.path.abspath(os.path.join(output_folder, '..', 'normal_region'))
    nrfp = np.load(os.path.join(path2nr, 'nrfp.npy'))
    fp = np.load(os.path.join(path2nr, 'fp@0.5.npy'))
    frocs=calc_froc(result["recs"], nrfp)
    result_str += "froc:\n"
    for i, froc in enumerate(frocs):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i), froc
        )
        
    result_str += "avg:\n"
    for i, froc in enumerate(frocs):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i), (froc.item()+result["rec_conf_thres"][i].item()+fp.item())/3
        )

    if result["froc"] is not None:
        result_str += "froc:\n"
        for i, froc in enumerate(result["froc"]):
            if i == 0:  # skip background
                continue
            result_str += "{:<16}: {:.4f}\n".format(
                dataset.map_class_id_to_class_name(i), froc
            )
    if result["fps"] is not None:
        result_str += "fps:\n"
        for i, fps in enumerate(result["fps"]):
            if i == 0:  # skip background
                continue
            result_str += "{:<16}: {:.4f}\n".format(
                dataset.map_class_id_to_class_name(i), fps
            )
    result_str += "rec@0.5:\n"
    for i, rec_conf_thres in enumerate(result["rec_conf_thres"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i), rec_conf_thres
        )
                   
    logger.info(result_str)
    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str)
        np.save(os.path.join(output_folder, 'recalls.npy'), np.array(result["recs"]))
    
    return result
                 
def calc_froc(rec, nrfp):
    """By adjusting confidence threshold, we can get various versions of prediction array. 
    When the numbers of normal region false positives are 1, 2, 4, 8, 16, 32 , 
    FROC is the average recall of these different versions of predictions. 
    Args:
        rec (numpy.array): 
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric 
            (with corection in precision) for calculating average precision. 
            The default value is: obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """
    n_fg_class = rec.shape[1]
    froc = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if rec[:,l][0] is None:
            froc[l] = np.nan
            continue
        rec_t = rec[:,l][(nrfp==1) | (nrfp==2) | (nrfp==4) | (nrfp==8) | (nrfp==16) | (nrfp==32)]
        if rec_t.size>0:
            froc[l] = rec_t.sum()/rec_t.size
        else:
            print('no conf threshold make nrfp 1,2,4,8,16,32')
            froc[l] = rec[500,l]
        
    return froc

def eval_detection_ringcell(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False, pred_boxlists_nr=None):
    """Evaluate on histo dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec, rec_conf_thres = calc_detection_ringcell_prec_rec(
        gt_boxlists=gt_boxlists, pred_boxlists=pred_boxlists, iou_thresh=iou_thresh
    )
    
    # recalls with respect to conf_t
    recs=[]
    for conf_t in np.arange(0.0, 1.0, 0.001):
        _, _, rec_conf = calc_detection_ringcell_prec_rec(
        gt_boxlists=gt_boxlists, pred_boxlists=pred_boxlists, iou_thresh=iou_thresh, confidence_threshold = conf_t)
        recs.append(rec_conf)
    
    ap = calc_detection_ringcell_ap(prec, rec, use_07_metric=use_07_metric)
    
    # !!!!!!!!!!
    if pred_boxlists_nr is not None:
        froc = calc_detection_ringcell_froc(gt_boxlists, pred_boxlists, iou_thresh=iou_thresh, pred_boxlists_nr=pred_boxlists_nr)
        _, fps = calc_detection_ringcell_nrfp_fps(pred_boxlists_nr, confidence_threshold = iou_thresh)
    else:
        froc = None
        fps = None
    return {"recs": np.array(recs), "ap": ap, "map": np.nanmean(ap), "froc": froc, "rec_conf_thres": rec_conf_thres, "fps": fps}


def calc_detection_ringcell_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5, confidence_threshold = 0.5):
    """Calculate precision and recall based on evaluation code of Ring Cell dataset.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in Ring Cell detection Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        if pred_boxlist.has_field('objectness'):
            pred_score = pred_boxlist.get_field("objectness").numpy()
        else:
            pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    #!!!!!!!!!!!!!!!!!
    # add by ys
    rec_conf_thres = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]
        score_l = score_l[order]
        
        # calculate tp/fp at fix confidence_threshold
        not_match_mask_l = match_l == 0
        not_match_score_l = score_l[not_match_mask_l]
        fp_conf_thres = (not_match_score_l>confidence_threshold).sum()
        
        match_mask_l = match_l == 1
        match_score_l = score_l[match_mask_l]
        tp_conf_thres = (match_score_l>confidence_threshold).sum()
        
        if n_pos[l] > 0:
            rec_conf_thres[l] = tp_conf_thres / n_pos[l]
            
        # calculate tp/fp at list of confidence_threshold
        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
            
#     print('=======================================',len(prec[1]))

    return prec, rec, rec_conf_thres

def calc_detection_ringcell_nrfp_fps(pred_boxlists_nr = None, confidence_threshold = 0.5):
    """Evaluate on Normal region (negative images) dataset.
    Calculate precision and recall based on evaluation code of Ring Cell dataset.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in Ring Cell detection Challenge.
    nrfp (int): Normal region false positives
    Normal region false positives is the average number of false positive predictions in the negative images. 
    For evaluation FPs will be written as Max(100 – Normal region false positives, 0).
   """
    score = defaultdict(list)
    match = defaultdict(list)
    if pred_boxlists_nr is not None:
        n_imgs = len(pred_boxlists_nr)
        for pred_boxlist in pred_boxlists_nr:
            pred_bbox = pred_boxlist.bbox.numpy()
            pred_label = pred_boxlist.get_field("labels").numpy()
            pred_score = pred_boxlist.get_field("scores").numpy()
            for l in np.unique(pred_label.astype(int)):
                pred_mask_l = pred_label == l
                pred_bbox_l = pred_bbox[pred_mask_l]
                pred_score_l = pred_score[pred_mask_l]
                # sort by score
                order = pred_score_l.argsort()[::-1]
                pred_score_l = pred_score_l[order]
                score[l].extend(pred_score_l)
                # all the match are set to 0
                match[l].extend((0,) * pred_bbox_l.shape[0])


        n_fg_class = max(score.keys()) + 1

        nrfp = [None] * n_fg_class
        nrfp_conf_thres = [None] * n_fg_class

        fps = [None] * n_fg_class
        fps_conf_thres = [None] * n_fg_class

        for l in score.keys():
            score_l = np.array(score[l])
            match_l = np.array(match[l], dtype=np.int8)

            # calculate at fix confidence_threshold
            nrfp_conf_thres[l] = (score_l>confidence_threshold).sum()//n_imgs
            fps_conf_thres[l] = np.maximum(100 - nrfp_conf_thres[l], 0)

            # calculate at list of confidence_threshold
            order = score_l.argsort()[::-1]
            match_l = match_l[order]
            fp = np.cumsum(match_l == 0)
            nrfp[l] = fp//n_imgs
            fps[l] = np.maximum(100 - nrfp[l], 0)

            # no need nrfp and fps now
    else:
        nrfp_conf_thres = None
        fps_conf_thres = None

    return nrfp_conf_thres, fps_conf_thres

def calc_detection_ringcell_froc(gt_boxlists, pred_boxlists, iou_thresh=0.5, pred_boxlists_nr=None):
    """By adjusting confidence threshold, we can get various versions of prediction array. 
    When the numbers of normal region false positives are 1, 2, 4, 8, 16, 32 , 
    FROC is the average recall of these different versions of predictions. 
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric 
            (with corection in precision) for calculating average precision. 
            The default value is: obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """
    nrfp = []
    rec = []
    if pred_boxlists_nr is not None:
        for conf_t in np.arange(0.0, 1.0, 0.001):
            nrfp_conf_thres, _ = calc_detection_ringcell_nrfp_fps(pred_boxlists_nr, confidence_threshold = conf_t)
            nrfp.append(nrfp_conf_thres)
            _, _, rec_conf_thres = calc_detection_histo_prec_rec(gt_boxlists, pred_boxlists, iou_thresh = iou_thresh, confidence_threshold = conf_t)
            rec.append(rec_conf_thres)
        
        nrfp = np.array(nrfp)
        rec = np.array(rec)

        n_fg_class = rec.shape[1]
        froc = np.empty(n_fg_class)
        for l in range(n_fg_class):
            if nrfp[:,l][0] is None or rec[:,l][0] is None:
                froc[l] = np.nan
                continue
            rec_t = rec[:,l][(nrfp[:,l]==1) | (nrfp[:,l]==2) | (nrfp[:,l]==4) | (nrfp[:,l]==8) | (nrfp[:,l]==16) | (nrfp[:,l]==32)]
            froc[l] = rec_t.sum()/rec_t.size
    else:
        froc = None
        
    return froc

def calc_detection_ringcell_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def calc_detection_ringcell_ar(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average recall
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric 
            (with corection in precision) for calculating average precision. 
            The default value is: obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(rec)
    ar = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ar[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ar[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(prec[l] >= t) == 0:
                    r = 0
                else:
                    r = np.max(np.nan_to_num(rec[l])[prec[l] >= t])
                ar[l] += r / 11
        else:
            # correct AR calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], prec[l], [0]))
            mrec = np.concatenate(([0], np.nan_to_num(rec[l]), [1]))

            mrec = np.maximum.accumulate(mrec[::-1])[::-1]

            # to calculate area under RP curve, look for points
            # where X axis (precision) changes value
            i = np.where(mpre[1:] != mpre[:-1])[0]

            # and sum (\Delta recall) * prec
            ar[l] = np.sum((mpre[i + 1] - mpre[i]) * mrec[i + 1])

    return ar
