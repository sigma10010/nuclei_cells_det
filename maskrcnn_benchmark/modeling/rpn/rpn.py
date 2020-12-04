# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.rpn.retinanet.retinanet import build_retinanet
from .loss import make_rpn_loss_evaluator
from .anchor_generator import make_anchor_generator
from .inference import make_rpn_postprocessor


class RPNHeadConvRegressor(nn.Module):
    """
    A simple RPN Head for classification and bbox regression
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted
        """
        super(RPNHeadConvRegressor, self).__init__()
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        logits = [self.cls_logits(y) for y in x]
        bbox_reg = [self.bbox_pred(y) for y in x]

        return logits, bbox_reg


class RPNHeadFeatureSingleConv(nn.Module):
    """
    Adds a simple RPN Head with one conv to extract the feature
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
        """
        super(RPNHeadFeatureSingleConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

        self.out_channels = in_channels

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        x = [F.relu(self.conv(z)) for z in x]

        return x
    
@registry.RPN_HEADS.register("EmbeddingConvRPNHead")
class EmbeddingRPNHead(nn.Module):
    """
    Adds a Embedding RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors, dim_embeddings = 12):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted (per location)
            dim_embeddings (int): number of dimention to be embedded
        """
        super(EmbeddingRPNHead, self).__init__()
        #----------architecture explaination-----------------------
        # cls head: conv-relu-embed-conv
        # shape: in_channels*H*W --> in_channels*H'*W' --> (num_anchors*dim_embeddings)*H'*W'--> num_anchors*H'*W'
        # box head: conv-relu-conv
        # shape: in_channels*H*W --> in_channels*H'*W' --> num_anchors*H'*W'
        # set output channels as num_anchors/num_anchors * 4
        # 
        #----------------------------------------------------------
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        ) 
        self.cls_embedding = nn.Conv2d(in_channels, num_anchors * dim_embeddings, kernel_size=1, stride=1)
        self.cls_logits = nn.Conv2d(num_anchors * dim_embeddings, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.cls_embedding, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            # cls follow on embedding
            t1 = self.cls_embedding(t)
            logits.append(self.cls_logits(t1))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg
    
    def get_embeddings(self, x):
        embeddings = []
        for feature in x:
            t = F.relu(self.conv(feature))
            embeddings.append(self.cls_embedding(t))
        return embeddings
    
@registry.RPN_HEADS.register("SingleConvRPNHead")
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    """

    def __init__(self, cfg, in_channels, num_anchors):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted (per location?)
        """
        super(RPNHead, self).__init__()
        #----------architecture explaination-----------------------
        # conv-relu-conv
        # shape: in_channels*H*W --> in_channels*H'*W' --> num_anchors*H'*W'
        # set output channels as num_anchors/num_anchors * 4
        # 
        #----------------------------------------------------------
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        ) 
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_logits, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

@registry.RPN_EMBEDS.register("SingleConvRPNEmbed")
class RPNEmbed(nn.Module):
    """
    Adds a simple RPN embed with embedding and regression heads.
    Use as an independent branch.
    """

    def __init__(self, cfg, in_channels, num_anchors, dim_embeddings = 12):
        """
        Arguments:
            cfg              : config
            in_channels (int): number of channels of the input feature
            num_anchors (int): number of anchors to be predicted (per location)
            dim_embeddings (int): number of dimention to be embedded
        """
        super(RPNEmbed, self).__init__()
        #----------architecture explaination-----------------------
        # conv-relu-conv
        # shape: in_channels*H*W --> in_channels*H'*W' --> num_anchors*H'*W'
        # set output channels as num_anchors/num_anchors * 4
        # (in this way to make sure that features extracting from each Anchor are of the same shape?)
        #----------------------------------------------------------
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        ) 
        self.cls_embedding = nn.Conv2d(in_channels, num_anchors * dim_embeddings, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )

        for l in [self.conv, self.cls_embedding, self.bbox_pred]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        embeddings = []
        bbox_reg = []
        for feature in x:
            t = F.relu(self.conv(feature))
            embeddings.append(self.cls_embedding(t))
            bbox_reg.append(self.bbox_pred(t))
        return embeddings, bbox_reg

class RPNModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and outputs 
    RPN proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg, in_channels, dim_embeddings = 12):
        """loss_mode (int):
        0: original_rpn
        1: siamese_rpn
        2: triplet_rpn
        """
        super(RPNModule, self).__init__()

        self.cfg = cfg.clone()
        
        anchor_generator = make_anchor_generator(cfg)

        rpn_head = registry.RPN_HEADS[cfg.MODEL.RPN.RPN_HEAD]
        if cfg.MODEL.RPN.RPN_HEAD == "SingleConvRPNHead":
            self.loss_mode = 0
            head = rpn_head(
                cfg, in_channels, anchor_generator.num_anchors_per_location()[0]
            )
        elif cfg.MODEL.RPN.RPN_HEAD == "EmbeddingConvRPNHead":
            self.loss_mode = cfg.MODEL.RPN.RPN_EMBED_LOSS
            self.dim_embeddings = cfg.MODEL.RPN.RPN_EMBED_DIM
            head = rpn_head(
                cfg, in_channels, anchor_generator.num_anchors_per_location()[0], dim_embeddings = self.dim_embeddings
            )
        else:
            raise RuntimeError("RPN HEAD not available: {}".format(cfg.MODEL.RPN.RPN_HEAD))
            

        rpn_box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        box_selector_train = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=True)
        box_selector_test = make_rpn_postprocessor(cfg, rpn_box_coder, is_train=False)

        loss_evaluator = make_rpn_loss_evaluator(cfg, rpn_box_coder, self.loss_mode)

        self.anchor_generator = anchor_generator
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        objectness, rpn_box_regression = self.head(features)
        anchors = self.anchor_generator(images, features)
#         embeddings = self.head.get_embeddings(features)

        if self.training:
            if self.cfg.MODEL.RPN.RPN_HEAD == "SingleConvRPNHead":
                return self._forward_train(anchors, objectness, rpn_box_regression, targets)
            elif self.cfg.MODEL.RPN.RPN_HEAD == "EmbeddingConvRPNHead":
                return self._forward_train(anchors, objectness, rpn_box_regression, targets, embeddings = self.head.get_embeddings(features))
        else:
            return self._forward_test(anchors, objectness, rpn_box_regression)

    def _forward_train(self, anchors, objectness, rpn_box_regression, targets, embeddings = None):
        if self.cfg.MODEL.RPN_ONLY:
            # When training an RPN-only model, the loss is determined by the
            # predicted objectness and rpn_box_regression values and there is
            # no need to transform the anchors into predicted boxes; this is an
            # optimization that avoids the unnecessary transformation.
            boxes = anchors
        else:
            # For end-to-end models, anchors must be transformed into boxes and
            # sampled into a training batch.
            with torch.no_grad():
                boxes = self.box_selector_train(
                    anchors, objectness, rpn_box_regression, targets
                )
        if self.loss_mode == 0:
            loss_objectness, loss_rpn_box_reg = self.loss_evaluator(
                anchors, objectness, rpn_box_regression, targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
            return boxes, losses
        else:
            loss_objectness, loss_rpn_box_reg, loss_embedding = self.loss_evaluator(
                anchors, objectness, rpn_box_regression, targets, embeddings
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
                "loss_embedding": loss_embedding,
            }
            return boxes, losses
        raise RuntimeError("RPN HEAD not available: {}".format(self.cfg.MODEL.RPN.RPN_HEAD))
        

    def _forward_test(self, anchors, objectness, rpn_box_regression):
        boxes = self.box_selector_test(anchors, objectness, rpn_box_regression)
        if self.cfg.MODEL.RPN_ONLY:
            # For end-to-end models, the RPN proposals are an intermediate state
            # and don't bother to sort them in decreasing score order. For RPN-only
            # models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            inds = [
                box.get_field("objectness").sort(descending=True)[1] for box in boxes
            ]
            boxes = [box[ind] for box, ind in zip(boxes, inds)]
        return boxes, {}


def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.RETINANET_ON:
        return build_retinanet(cfg, in_channels)

    return RPNModule(cfg, in_channels)
