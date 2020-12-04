"""
This file contains specific functions for computing losses on the RetinaNet
file
"""
import numpy as np
import torch
from torch.nn import functional as F

from ..utils import concat_box_prediction_layers, concat_box_prediction_embeddings_layers

from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import BalancedPositiveNegativeSampler

from maskrcnn_benchmark.layers import ContrastiveLoss, TripletLoss
from maskrcnn_benchmark.layers import pair_embeddings, triplet_embeddings
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.rpn.loss import RPNLossComputation

class RetinaNetLossComputation(RPNLossComputation):
    """
    This class computes the RetinaNet loss.
    """
    TRIPLET_MARGIN = 1
    PAIR_MARGIN = 1
    def __init__(self, proposal_matcher, box_coder,
                 generate_labels_func,
                 sigmoid_focal_loss,
                 fg_bg_sampler,
                 bbox_reg_beta=0.11,
                 embed_margin=1.0,
                 embedding_loss = 2,
                 regress_norm=1.0):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.box_coder = box_coder
        self.box_cls_loss_func = sigmoid_focal_loss
        self.bbox_reg_beta = bbox_reg_beta
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        self.regress_norm = regress_norm
        self.fg_bg_sampler = fg_bg_sampler
        self.embed_margin = embed_margin
        self.embedding_loss = embedding_loss

    def __call__(self, anchors, box_cls, box_regression, targets, embeddings = None):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            retinanet_cls_loss (Tensor)
            retinanet_regression_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        N = len(labels)
        if embeddings is not None:
            box_cls, box_regression, embeddings = \
                concat_box_prediction_embeddings_layers(box_cls, box_regression, embeddings)
        else:
            box_cls, box_regression = \
                    concat_box_prediction_layers(box_cls, box_regression)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)

        retinanet_regression_loss = smooth_l1_loss(
            box_regression[pos_inds],
            regression_targets[pos_inds],
            beta=self.bbox_reg_beta,
            size_average=False,
        ) / (max(1, pos_inds.numel() * self.regress_norm))

        labels = labels.int()

        retinanet_cls_loss = self.box_cls_loss_func(
            box_cls,
            labels
        ) / (pos_inds.numel() + N)
        
        # triplet loss
        if embeddings is not None and self.embedding_loss == 2:
            margin = self.embed_margin
#             print('triplet margin:', margin)
            T_Loss = TripletLoss(margin)
            # hard negtive mining version
            anchor_embeddings, positive_embeddings, negative_embeddings = triplet_embeddings(embeddings[sampled_inds], labels[sampled_inds])
#             anchor_embeddings, positive_embeddings, negative_embeddings = triplet_embeddings(embeddings, labels)
            triplet_loss = T_Loss(anchor_embeddings, positive_embeddings, negative_embeddings, size_average=True)
            # dynamic incremental margin
#             if triplet_loss == 0 and np.random.random() > 0.5:
#                 RetinaNetLossComputation.TRIPLET_MARGIN += 1
            return retinanet_cls_loss, retinanet_regression_loss, triplet_loss

        # pair loss
        elif embeddings is not None and self.embedding_loss == 1:
            # print('pair loss ===============================')
            margin = self.embed_margin
            C_loss = ContrastiveLoss(margin)
            embeddings1, embeddings2, targets = pair_embeddings(embeddings[sampled_inds], labels[sampled_inds])
            pair_loss = C_loss(embeddings1, embeddings2, targets)
            return retinanet_cls_loss, retinanet_regression_loss, pair_loss
        
        else:
            return retinanet_cls_loss, retinanet_regression_loss


def generate_retinanet_labels(matched_targets):
    labels_per_image = matched_targets.get_field("labels")
    return labels_per_image


def make_retinanet_loss_evaluator(cfg, box_coder):
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )
    matcher = Matcher(
        cfg.MODEL.RETINANET.FG_IOU_THRESHOLD,
        cfg.MODEL.RETINANET.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )
    sigmoid_focal_loss = SigmoidFocalLoss(
        cfg.MODEL.RETINANET.LOSS_GAMMA,
        cfg.MODEL.RETINANET.LOSS_ALPHA
    )
    
    loss_evaluator = RetinaNetLossComputation(
        matcher,
        box_coder,
        generate_retinanet_labels,
        sigmoid_focal_loss,
        fg_bg_sampler,
        bbox_reg_beta = cfg.MODEL.RETINANET.BBOX_REG_BETA,
        embed_margin = cfg.MODEL.RETINANET.EMBED_MARGIN,
        embedding_loss = cfg.MODEL.RETINANET.EMBED_LOSS,
        regress_norm = cfg.MODEL.RETINANET.BBOX_REG_WEIGHT
    )
    return loss_evaluator
