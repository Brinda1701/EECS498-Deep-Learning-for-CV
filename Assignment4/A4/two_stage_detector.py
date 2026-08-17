import math
from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from a4_helper import *
from common import class_spec_nms, get_fpn_location_coords, nms
from torch import nn
from torch.nn import functional as F

# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]


def hello_two_stage_detector():
    print("Hello from two_stage_detector.py!")


class RPNPredictionNetwork(nn.Module):
    """
    RPN prediction network that accepts FPN feature maps from different levels
    and makes two predictions for every anchor: objectness and box deltas.

    Faster R-CNN typically uses (p2, p3, p4, p5) feature maps. We will exclude
    p2 for have a small enough model for Colab.

    Conceptually this module is quite similar to `FCOSPredictionNetwork`.
    """

    def __init__(
        self, in_channels: int, stem_channels: List[int], num_anchors: int = 3
    ):
        """
        Args:
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
            num_anchors: Number of anchor boxes assumed per location (say, `A`).
                Faster R-CNN without an FPN uses `A = 9`, anchors with three
                different sizes and aspect ratios. With FPN, it is more common
                to have a fixed size dependent on the stride of FPN level, hence
                `A = 3` is default - with three aspect ratios.
        """
        super().__init__()

        self.num_anchors = num_anchors
        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules. RPN shares this stem for objectness and box
        # regression (unlike FCOS, that uses separate stems).
        #
        # Use `in_channels` and `stem_channels` for creating these layers, the
        # docstring above tells you what they mean. Initialize weights of each
        # conv layer from a normal distribution with mean = 0 and std dev = 0.01
        # and all biases with zero. Use conv stride = 1 and zero padding such
        # that size of input features remains same: remember we need predictions
        # at every location in feature map, we shouldn't "lose" any locations.
        ######################################################################
        # Fill this list. It is okay to use your implementation from
        # `FCOSPredictionNetwork` for this code block.
        stem_rpn = []
        cur_in_channels = in_channels
        # 构建主干卷积层：由 3x3 卷积 + ReLU 组成
        for out_c in stem_channels:
            conv = nn.Conv2d(cur_in_channels, out_c, kernel_size=3, stride=1, padding=1)
            # 使用均值 0，标准差 0.01 的正态分布初始化权重，偏置初始化为 0
            nn.init.normal_(conv.weight, mean=0.0, std=0.01)
            nn.init.constant_(conv.bias, 0.0)

            stem_rpn.append(conv)
            stem_rpn.append(nn.ReLU())

            cur_in_channels = out_c
        # 封装为Sequential模块
        self.stem_rpn = nn.Sequential(*stem_rpn)
        ######################################################################
        # TODO: Create TWO 1x1 conv layers for individually to predict
        # objectness and box deltas for every anchor, at every location.
        #
        # Objectness is obtained by applying sigmoid to its logits. However,
        # DO NOT initialize a sigmoid module here. PyTorch loss functions have
        # numerically stable implementations with logits.
        ######################################################################

        # 1x1 卷积预测每个 anchor 的前景/背景概率 (objectness logits)
        self.pred_obj = nn.Conv2d(stem_channels[-1], self.num_anchors, kernel_size=1, stride=1, padding=0)
        # 1x1 卷积预测每个 anchor 的 4 个边界框回归偏移量 (box deltas)
        self.pred_box = nn.Conv2d(stem_channels[-1], self.num_anchors * 4, kernel_size=1, stride=1, padding=0)

        # 初始化预测层的权重和偏置
        nn.init.normal_(self.pred_obj.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.pred_box.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.pred_obj.bias, 0.0)
        nn.init.constant_(self.pred_box.bias, 0.0)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict desired quantities for every anchor
        at every location. Format the output tensors such that feature height,
        width, and number of anchors are collapsed into a single dimension (see
        description below in "Returns" section) this is convenient for computing
        loss and perforning inference.

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}.
                Each tensor will have shape `(batch_size, fpn_channels, H, W)`.

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Objectness logits:     `(batch_size, H * W * num_anchors)`
            2. Box regression deltas: `(batch_size, H * W * num_anchors, 4)`
        """

        ######################################################################
        # TODO: Iterate over every FPN feature map and obtain predictions using
        # the layers defined above. DO NOT apply sigmoid to objectness logits.
        ######################################################################
        object_logits = {}
        boxreg_deltas = {}

        # 遍历每个 FPN 层级 (p3, p4, p5) 提取预测结果
        for level_name, feats in feats_per_fpn_level.items():
            # 1. 通过 RPN 的主干 (stem) 卷积网络
            feats = self.stem_rpn(feats)
            N, _, H, W = feats.shape

            # 2. 预测前景/背景 (Objectness) Logits
            # pred_obj 输出 shape: (N, num_anchors, H, W)
            # permute 为 (N, H, W, num_anchors)，再展平为 (N, H * W * num_anchors)
            obj = self.pred_obj(feats)
            obj = obj.permute(0, 2, 3, 1).reshape(N, -1)
            object_logits[level_name] = obj

            # 3. 预测边界框回归偏移量 (Box Regression Deltas)
            # pred_box 输出 shape: (N, num_anchors * 4, H, W)
            # 先 reshape 为 (N, num_anchors, 4, H, W)，再 permute 为 (N, H, W, num_anchors, 4)
            # 最后展平为 (N, H * W * num_anchors, 4)
            box = self.pred_box(feats)
            box = box.reshape(N, self.num_anchors, 4, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, 4)
            boxreg_deltas[level_name] = box
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return [object_logits, boxreg_deltas]


@torch.no_grad()
def generate_fpn_anchors(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    stride_scale: int,
    aspect_ratios: List[float] = [0.5, 1.0, 2.0],
):
    """
    Generate multiple anchor boxes at every location of FPN level. Anchor boxes
    should be in XYXY format and they should be centered at the given locations.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H, W is the size of FPN feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `common.py` for more details.
        stride_scale: Size of square anchor at every FPN levels will be
            `(this value) * (FPN level stride)`. Default is 4, which will make
            anchor boxes of size (32x32), (64x64), (128x128) for FPN levels
            p3, p4, and p5 respectively.
        aspect_ratios: Anchor aspect ratios to consider at every location. We
            consider anchor area to be `(stride_scale * FPN level stride) ** 2`
            and set new width and height of anchors at every location:
                new_width = sqrt(area / aspect ratio)
                new_height = area / new_width

    Returns:
        TensorDict
            Dictionary with same keys as `locations_per_fpn_level` and values as
            tensors of shape `(HWA, 4)` giving anchors for all locations
            per FPN level, each location having `A = len(aspect_ratios)` anchors.
            All anchors are in XYXY format and their centers align with locations.
    """

    # Set these to `(N, A, 4)` Tensors giving anchor boxes in XYXY format.
    anchors_per_fpn_level = {
        level_name: None for level_name, _ in locations_per_fpn_level.items()
    }

    for level_name, locations in locations_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        # List of `A = len(aspect_ratios)` anchor boxes.
        anchor_boxes = []
        for aspect_ratio in aspect_ratios:
            ##################################################################
            # TODO: Implement logic for anchor boxes below. Write vectorized
            # implementation to generate anchors for a single aspect ratio.
            # Fill `anchor_boxes` list above.
            #
            # Calculate resulting width and height of the anchor box as per
            # `stride_scale` and `aspect_ratios` definitions. Then shift the
            # locations to get top-left and bottom-right co-ordinates.
            ##################################################################
            # Replace "pass" statement with your code
            # 基础锚框尺寸基准：stride_scale * level_stride
            # 例如 p3 (stride 8, scale 4 -> 32x32)
            size = torch.tensor(stride_scale * level_stride, dtype=locations.dtype, device=locations.device)
            area = size ** 2
            # 根据宽高比 aspect_ratio 计算新的宽度和高度
            # new_width = sqrt(area / aspect_ratio), new_height = area / new_width
            new_width = torch.sqrt(area / aspect_ratio)
            new_height = area / new_width

            # 以特征图中心 locations (center_x, center_y) 为中心，计算左上角 (x1, y1) 和右下角 (x2, y2)
            x1 = locations[:, 0] - new_width / 2
            y1 = locations[:, 1] - new_height / 2
            x2 = locations[:, 0] + new_width / 2
            y2 = locations[:, 1] + new_height / 2
            anchor_boxes.append(torch.stack([x1, y1, x2, y2], dim=1))
            ##################################################################
            #                           END OF YOUR CODE                     #
            ##################################################################

        # 堆叠各个宽高比的 anchor，shape: (A, H * W, 4)
        anchor_boxes = torch.stack(anchor_boxes)
        # 转置为 (H * W, A, 4)，然后展平为 (H * W * A, 4)，保证同一位置的不同 anchor 在内存中连续
        anchor_boxes = anchor_boxes.permute(1, 0, 2).contiguous().view(-1, 4)
        anchors_per_fpn_level[level_name] = anchor_boxes

    return anchors_per_fpn_level


@torch.no_grad()
def iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute intersection-over-union (IoU) between pairs of box tensors. Input
    box tensors must in XYXY format.

    Args:
        boxes1: Tensor of shape `(M, 4)` giving a set of box co-ordinates.
        boxes2: Tensor of shape `(N, 4)` giving another set of box co-ordinates.

    Returns:
        torch.Tensor
            Tensor of shape (M, N) with `iou[i, j]` giving IoU between i-th box
            in `boxes1` and j-th box in `boxes2`.
    """

    ##########################################################################
    # TODO: Implement the IoU function here.                                 #
    ##########################################################################
    # 1. 计算两组框各自的面积: (width * height)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 2. 利用广播机制计算相交区域的左上角 (x1, y1) 和右下角 (x2, y2)
    # boxes1: (M, 4) -> (M, 1, 4), boxes2: (N, 4)
    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    # 3. 计算交集面积 (若不相交，width 或 height < 0，使用 clamp 归零)
    inter = (torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0))

    # 4. 计算并集面积及 IoU
    union = area1[:, None] + area2 - inter
    iou = inter / union
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return iou


@torch.no_grad()
def rcnn_match_anchors_to_gt(
    anchor_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresholds: Tuple[float, float],
) -> TensorDict:
    """
    Match anchor boxes (or RPN proposals) with a set of GT boxes. Anchors having
    high IoU with any GT box are assigned "foreground" and matched with that box
    or vice-versa.

    NOTE: This function is NOT BATCHED. Call separately for GT boxes per image.

    Args:
        anchor_boxes: Anchor boxes (or RPN proposals). Dictionary of three keys
            a combined tensor of some shape `(N, 4)` where `N` are total anchors
            from all FPN levels, or a set of RPN proposals.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.
        iou_thresholds: Tuple of (low, high) IoU thresholds, both in [0, 1]
            giving thresholds to assign foreground/background anchors.
    """

    # Filter empty GT boxes:
    gt_boxes = gt_boxes[gt_boxes[:, 4] != -1]

    # If no GT boxes are available, match all anchors to background and return.
    if len(gt_boxes) == 0:
        fake_boxes = torch.zeros_like(anchor_boxes) - 1
        fake_class = torch.zeros_like(anchor_boxes[:, [0]]) - 1
        return torch.cat([fake_boxes, fake_class], dim=1)

    # Match matrix => pairwise IoU of anchors (rows) and GT boxes (columns).
    # STUDENTS: This matching depends on your IoU implementation.
    match_matrix = iou(anchor_boxes, gt_boxes[:, :4])

    # Find matched ground-truth instance per anchor:
    match_quality, matched_idxs = match_matrix.max(dim=1)
    matched_gt_boxes = gt_boxes[matched_idxs]

    # Set boxes with low IoU threshold to background (-1).
    matched_gt_boxes[match_quality <= iou_thresholds[0]] = -1

    # Set remaining boxes to neutral (-1e8).
    neutral_idxs = (match_quality > iou_thresholds[0]) & (
        match_quality < iou_thresholds[1]
    )
    matched_gt_boxes[neutral_idxs, :] = -1e8
    return matched_gt_boxes


def rcnn_get_deltas_from_anchors(
    anchors: torch.Tensor, gt_boxes: torch.Tensor
) -> torch.Tensor:
    """
    Get box regression deltas that transform `anchors` to `gt_boxes`. These
    deltas will become GT targets for box regression. Unlike FCOS, the deltas
    are in `(dx, dy, dw, dh)` format that represent offsets to anchor centers
    and scaling factors for anchor size. Box regression is only supervised by
    foreground anchors. If GT boxes are "background/neutral", then deltas
    must be `(-1e8, -1e8, -1e8, -1e8)` (just some LARGE negative number).

    Follow Slide 68:
        https://web.eecs.umich.edu/~justincj/slides/eecs498/WI2022/598_WI2022_lecture13.pdf

    Args:
        anchors: Tensor of shape `(N, 4)` giving anchors boxes in XYXY format.
        gt_boxes: Tensor of shape `(N, 4)` giving matching GT boxes.

    Returns:
        torch.Tensor
            Tensor of shape `(N, 4)` giving anchor deltas.
    """
    ##########################################################################
    # TODO: Implement the logic to get deltas.                               #
    # Remember to set the deltas of "background/neutral" GT boxes to -1e8    #
    ##########################################################################
    # 1. 计算 anchor 的宽度 wa、高度 ha 和中心坐标 (xa, ya)
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    xa = (anchors[:, 0] + anchors[:, 2]) / 2
    ya = (anchors[:, 1] + anchors[:, 3]) / 2

    # 2. 计算匹配的 GT box 的宽度 w_gt、高度 h_gt 和中心坐标 (x_gt, y_gt)
    w_gt = gt_boxes[:, 2] - gt_boxes[:, 0]
    h_gt = gt_boxes[:, 3] - gt_boxes[:, 1]
    x_gt = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    y_gt = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2

    # 3. 遵循 Faster R-CNN 变换公式计算 (dx, dy, dw, dh):
    # dx = (x_gt - xa) / wa,  dy = (y_gt - ya) / ha
    # dw = log(w_gt / wa),     dh = log(h_gt / ha)
    dx = (x_gt - xa) / wa
    dy = (y_gt - ya) / ha
    dw = torch.log(torch.clamp(w_gt, min=1e-5) / wa)
    dh = torch.log(torch.clamp(h_gt, min=1e-5) / ha)

    deltas = torch.stack([dx, dy, dw, dh], dim=1)

    # 4. 对于背景/中性锚框 (GT 坐标为负值 -1 或 -1e8)，将 delta 设为 -1e8
    bg_mask = (gt_boxes[:, 0] < 0) | (gt_boxes[:, 1] < 0) | (gt_boxes[:, 2] < 0) | (gt_boxes[:, 3] < 0)
    deltas[bg_mask] = -1e8
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return deltas


def rcnn_apply_deltas_to_anchors(
    deltas: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    """
    Implement the inverse of `rcnn_get_deltas_from_anchors` here.

    Args:
        deltas: Tensor of shape `(N, 4)` giving box regression deltas.
        anchors: Tensor of shape `(N, 4)` giving anchors to apply deltas on.

    Returns:
        torch.Tensor
            Same shape as deltas and locations, giving the resulting boxes in
            XYXY format.
    """

    # Clamp dw and dh such that they would transform a 8px box no larger than
    # 224px. This is necessary for numerical stability as we apply exponential.
    scale_clamp = math.log(224 / 8)
    deltas = deltas.clone()
    deltas[:, 2] = torch.clamp(deltas[:, 2], max=scale_clamp)
    deltas[:, 3] = torch.clamp(deltas[:, 3], max=scale_clamp)

    ##########################################################################
    # TODO: Implement the transformation logic to get output boxes.          #
    ##########################################################################
    # 1. 计算 anchor 的宽度、高度及中心点 (xa, ya)
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_centers_x = (anchors[:, 0] + anchors[:, 2]) / 2
    anchor_centers_y = (anchors[:, 1] + anchors[:, 3]) / 2

    # 2. 逆变换：根据公式由 deltas 计算预测的目标框中心 (pred_cx, pred_cy) 和尺寸 (pred_w, pred_h)
    pred_centers_x = anchor_widths * deltas[:, 0] + anchor_centers_x
    pred_centers_y = anchor_heights * deltas[:, 1] + anchor_centers_y
    pred_widths = anchor_widths * torch.exp(deltas[:, 2])
    pred_heights = anchor_heights * torch.exp(deltas[:, 3])

    # 3. 由中心点和尺寸转换回 XYXY 格式 (x1, y1, x2, y2)
    output_boxes = torch.stack([
        pred_centers_x - pred_widths / 2,
        pred_centers_y - pred_heights / 2,
        pred_centers_x + pred_widths / 2,
        pred_centers_y + pred_heights / 2
    ], dim=1)
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return output_boxes


@torch.no_grad()
def sample_rpn_training(
    gt_boxes: torch.Tensor, num_samples: int, fg_fraction: float
):
    """
    Return `num_samples` (or fewer, if not enough found) random pairs of anchors
    and GT boxes without exceeding `fg_fraction * num_samples` positives, and
    then try to fill the remaining slots with background anchors. We will ignore
    "neutral" anchors in this sampling as they are not used for training.

    Args:
        gt_boxes: Tensor of shape `(N, 5)` giving GT box co-ordinates that are
            already matched with some anchor boxes (with GT class label at last
            dimension). Label -1 means background and -1e8 means meutral.
        num_samples: Total anchor-GT pairs with label >= -1 to return.
        fg_fraction: The number of subsampled labels with values >= 0 is
            `min(num_foreground, int(fg_fraction * num_samples))`. In other
            words, if there are not enough fg, the sample is filled with
            (duplicate) bg.

    Returns:
        fg_idx, bg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or
            fewer. Use these to index anchors, GT boxes, and model predictions.
    """
    foreground = (gt_boxes[:, 4] >= 0).nonzero().squeeze(1)
    background = (gt_boxes[:, 4] == -1).nonzero().squeeze(1)

    # Protect against not enough foreground examples.
    num_fg = min(int(num_samples * fg_fraction), foreground.numel())
    num_bg = num_samples - num_fg

    # Randomly select positive and negative examples.
    perm1 = torch.randperm(foreground.numel(), device=foreground.device)[:num_fg]
    perm2 = torch.randperm(background.numel(), device=background.device)[:num_bg]

    fg_idx = foreground[perm1]
    bg_idx = background[perm2]
    return fg_idx, bg_idx


@torch.no_grad()
def mix_gt_with_proposals(
    proposals_per_fpn_level: Dict[str, List[torch.Tensor]], gt_boxes: torch.Tensor
):
    """
    At start of training, RPN proposals may be low quality. It's possible that
    very few of these have high IoU with GT boxes. This may stall or de-stabilize
    training of second stage. This function mixes GT boxes with RPN proposals to
    improve training. Different GT boxes are mixed with proposals from different
    FPN levels according to assignment rule of FPN paper.

    Args:
        proposals_per_fpn_level: Dict of proposals per FPN level, per image in
            batch. These are same as outputs from `RPN.forward()` method.
        gt_boxes: Tensor of shape `(B, M, 4 or 5)` giving GT boxes per image in
            batch (with or without GT class label, doesn't matter).

    Returns:
        proposals_per_fpn_level: Same as input, but with GT boxes mixed in them.
    """

    # Mix ground-truth boxes for every example, per FPN level. There's no direct
    # way to vectorize this.
    for _idx, _gtb in enumerate(gt_boxes):

        # Filter empty GT boxes:
        _gtb = _gtb[_gtb[:, 4] != -1]
        if len(_gtb) == 0:
            continue

        # Compute FPN level assignments for each GT box. This follows Equation (1)
        # of FPN paper (k0 = 5). `level_assn` has `(M, )` integers, one of {3,4,5}
        _gt_area = (_gtb[:, 2] - _gtb[:, 0]) * (_gtb[:, 3] - _gtb[:, 1])
        level_assn = torch.floor(5 + torch.log2(torch.sqrt(_gt_area) / 224))
        level_assn = torch.clamp(level_assn, min=3, max=5).to(torch.int64)

        for level_name, _props in proposals_per_fpn_level.items():
            _prop = _props[_idx]

            # Get GT boxes of this image that match level scale, and append them
            # to proposals.
            _gt_boxes_fpn_subset = _gtb[level_assn == int(level_name[1])]
            if len(_gt_boxes_fpn_subset) > 0:
                proposals_per_fpn_level[level_name][_idx] = torch.cat(
                    # Remove class label since proposals don't have it:
                    [_prop, _gt_boxes_fpn_subset[:, :4]],
                    dim=0,
                )

    return proposals_per_fpn_level


class RPN(nn.Module):
    """
    Region Proposal Network: First stage of Faster R-CNN detector.

    This class puts together everything you implemented so far. It accepts FPN
    features as input and uses `RPNPredictionNetwork` to predict objectness and
    box reg deltas. Computes proposal boxes for second stage (during both
    training and inference) and losses during training.
    """

    def __init__(
        self,
        fpn_channels: int,
        stem_channels: List[int],
        batch_size_per_image: int = 16,
        anchor_stride_scale: int = 8,
        anchor_aspect_ratios: List[int] = [0.5, 1.0, 2.0],
        anchor_iou_thresholds: Tuple[int, int] = (0.3, 0.6),
        nms_thresh: float = 0.7,
        pre_nms_topk: int = 400,
        post_nms_topk: int = 100,
    ):
        """
        Args:
            batch_size_per_image: Anchors per image to sample for training.
            nms_thresh: IoU threshold for NMS - unlike FCOS, this is used
                during both, training and inference.
            pre_nms_topk: Number of top-K proposals to select before applying
                NMS, per FPN level. This helps in speeding up NMS.
            post_nms_topk: Number of top-K proposals to select after applying
                NMS, per FPN level. NMS is obviously going to be class-agnostic.

        Refer explanations of remaining args in the classes/functions above.
        """
        super().__init__()
        self.pred_net = RPNPredictionNetwork(
            fpn_channels, stem_channels, num_anchors=len(anchor_aspect_ratios)
        )
        # Record all input arguments:
        self.batch_size_per_image = batch_size_per_image
        self.anchor_stride_scale = anchor_stride_scale
        self.anchor_aspect_ratios = anchor_aspect_ratios
        self.anchor_iou_thresholds = anchor_iou_thresholds
        self.nms_thresh = nms_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk

    def forward(
        self,
        feats_per_fpn_level: TensorDict,
        strides_per_fpn_level: TensorDict,
        gt_boxes: Optional[torch.Tensor] = None,
    ):
        # Get batch size from FPN feats:
        num_images = feats_per_fpn_level["p3"].shape[0]

        ######################################################################
        # TODO: Implement the training forward pass. Follow these steps:
        #   1. Pass the FPN features per level to the RPN prediction network.
        #   2. Generate anchor boxes for all FPN levels.
        #
        # HINT: You have already implemented everything, just have to call the
        # appropriate functions.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        pred_obj_logits, pred_boxreg_deltas, anchors_per_fpn_level = (
            None,
            None,
            None,
        )
        # Replace "pass" statement with your code
        # 1. 将 FPN 特征输入 RPN 预测网络，获取各层级的 objectness logits 和 box regression deltas
        pred_obj_logits, pred_boxreg_deltas = self.pred_net(feats_per_fpn_level)

        # 2. 生成所有 FPN 层级的绝对像素中心坐标 (locations)
        locations_per_fpn_level = get_fpn_location_coords(
            {lvl: feat.shape for lvl, feat in feats_per_fpn_level.items()},
            strides_per_fpn_level=strides_per_fpn_level,
            dtype=feats_per_fpn_level["p3"].dtype,
            device=feats_per_fpn_level["p3"].device,
        )

        # 3. 生成所有 FPN 层级的 anchor boxes
        anchors_per_fpn_level = generate_fpn_anchors(
            locations_per_fpn_level,
            strides_per_fpn_level,
            self.anchor_stride_scale,
            self.anchor_aspect_ratios,
        )
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # We will fill three values in this output dict - "proposals",
        # "loss_rpn_box" (training only), "loss_rpn_obj" (training only)
        output_dict = {}

        # Get image height and width according to feature sizes and strides.
        # We need these to clamp proposals (These should be (224, 224) but we
        # avoid hard-coding them).
        img_h = feats_per_fpn_level["p3"].shape[2] * strides_per_fpn_level["p3"]
        img_w = feats_per_fpn_level["p3"].shape[3] * strides_per_fpn_level["p3"]

        # STUDENT: Implement this method before moving forward with the rest
        # of this `forward` method.
        output_dict["proposals"] = self.predict_proposals(
            anchors_per_fpn_level,
            pred_obj_logits,
            pred_boxreg_deltas,
            (img_w, img_h),
        )
        # Return here during inference - loss computation not required.
        if not self.training:
            return output_dict

        # ... otherwise continue loss computation:
        ######################################################################
        # Match the generated anchors with provided GT boxes. This
        # function is not batched so you may use a for-loop, like FCOS.
        ######################################################################
        # Combine anchor boxes from all FPN levels - we do not need any
        # distinction of boxes across different levels (for training).
        anchor_boxes = self._cat_across_fpn_levels(anchors_per_fpn_level, dim=0)

        # 针对 Batch 中的每张图片，将生成的 anchors 与真实标注框 (gt_boxes) 进行匹配
        matched_gt_boxes = []
        for i in range(num_images):
            matched_gt_boxes.append(
                rcnn_match_anchors_to_gt(
                    anchor_boxes, gt_boxes[i], self.anchor_iou_thresholds
                )
            )
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Combine matched boxes from all images to a `(B, HWA, 5)` tensor.
        matched_gt_boxes = torch.stack(matched_gt_boxes, dim=0)

        # Combine predictions across all FPN levels.
        pred_obj_logits = self._cat_across_fpn_levels(pred_obj_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)

        if self.training:
            # Repeat anchor boxes `batch_size` times so there is a 1:1
            # correspondence with GT boxes.
            anchor_boxes = anchor_boxes.unsqueeze(0).repeat(num_images, 1, 1)
            anchor_boxes = anchor_boxes.contiguous().view(-1, 4)

            # Collapse `batch_size`, and `HWA` to a single dimension so we have
            # simple `(-1, 4 or 5)` tensors. This simplifies loss computation.
            matched_gt_boxes = matched_gt_boxes.view(-1, 5)
            pred_obj_logits = pred_obj_logits.view(-1)
            pred_boxreg_deltas = pred_boxreg_deltas.view(-1, 4)

            ##################################################################
            # TODO: Compute training losses. Follow three steps in order:
            #   1. Sample a few anchor boxes for training. Pass the variable
            #      `matched_gt_boxes` to `sample_rpn_training` function and
            #      use those indices to get subset of predictions and targets.
            #      RPN samples 50-50% foreground/background anchors, unless
            #      there aren't enough foreground anchors.
            #
            #   2. Compute GT targets for box regression (you have implemented
            #      the transformation function already).
            #
            #   3. Calculate objectness and box reg losses per sampled anchor.
            #      Remember to set box loss for "background" anchors to 0.
            ##################################################################
            # 1. 采样用于训练的 anchor 索引 (前景和背景比例 50:50)
            total_batch_size = self.batch_size_per_image * num_images
            fg_idx, bg_idx = sample_rpn_training(
                matched_gt_boxes, total_batch_size, fg_fraction=0.5
            )

            # 2. 计算 Objectness 损失 (BCE Loss)
            # 提取采样出的前景和背景 logits
            sampled_obj_logits = pred_obj_logits[torch.cat([fg_idx, bg_idx])]
            # 构建二分类标签 (前景=1, 背景=0)
            sampled_obj_targets = torch.cat([
                torch.ones(len(fg_idx), device=pred_obj_logits.device),
                torch.zeros(len(bg_idx), device=pred_obj_logits.device),
            ])
            loss_obj = F.binary_cross_entropy_with_logits(
                sampled_obj_logits, sampled_obj_targets, reduction="none"
            )

            # 3. 计算 Box Regression 损失 (L1 Loss) - 仅由前景 anchor 监督
            sampled_pred_deltas = pred_boxreg_deltas[fg_idx]
            sampled_anchors = anchor_boxes[fg_idx]
            sampled_gt = matched_gt_boxes[fg_idx]
            gt_deltas = rcnn_get_deltas_from_anchors(sampled_anchors, sampled_gt[:, :4])

            loss_box = F.l1_loss(sampled_pred_deltas, gt_deltas, reduction="none")
            loss_box[gt_deltas == -1e8] *= 0.0
            ##################################################################
            #                         END OF YOUR CODE                       #
            ##################################################################

            # Sum losses and average by num(foreground + background) anchors.
            # In training code, we simply add these two and call `.backward()`
            output_dict["loss_rpn_obj"] = loss_obj.sum() / total_batch_size
            output_dict["loss_rpn_box"] = loss_box.sum() / total_batch_size

        return output_dict

    @torch.no_grad()  # Don't track gradients in this function.
    def predict_proposals(
        self,
        anchors_per_fpn_level: Dict[str, torch.Tensor],
        pred_obj_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        image_size: Tuple[int, int],  # (width, height)
    ):
        """
        Predict proposals for a batch of images for the second stage. Other
        input arguments are same as those computed in `forward` method. This
        method should not be called from anywhere except from inside `forward`.

        Returns:
            torch.Tensor
                proposals: Tensor of shape `(keep_topk, 4)` giving *absolute*
                    XYXY co-ordinates of predicted proposals. These will serve
                    as anchor boxes for the second stage.
        """

        # Gather proposals from all FPN levels in this list.
        proposals_all_levels = {
            level_name: None for level_name, _ in anchors_per_fpn_level.items()
        }
        for level_name in anchors_per_fpn_level.keys():

            # Get anchor boxes and predictions from a single level.
            level_anchors = anchors_per_fpn_level[level_name]

            # shape: (batch_size, HWA), (batch_size, HWA, 4)
            level_obj_logits = pred_obj_logits[level_name]
            level_boxreg_deltas = pred_boxreg_deltas[level_name]

            # Fill proposals per image, for this FPN level, in this list.
            level_proposals_per_image = []
            for _batch_idx in range(level_obj_logits.shape[0]):
                ##############################################################
                # TODO: Perform the following steps in order:
                #   1. Transform the anchors to proposal boxes using predicted
                #      box deltas, clamp to image height and width.
                #   2. Sort all proposals by their predicted objectness, and
                #      retain `self.pre_nms_topk` proposals. This speeds up
                #      our NMS computation. HINT: `torch.topk`
                #   3. Apply NMS and retain `keep_topk_per_level` proposals
                #      per image, per level.
                ##############################################################
                # 步骤 1: 将预测的 box deltas 应用到 anchors 上得到 proposals，并限制坐标在图像边界内
                proposals = rcnn_apply_deltas_to_anchors(
                    level_boxreg_deltas[_batch_idx], level_anchors
                )
                proposals[:, 0] = torch.clamp(proposals[:, 0], min=0, max=image_size[0])
                proposals[:, 1] = torch.clamp(proposals[:, 1], min=0, max=image_size[1])
                proposals[:, 2] = torch.clamp(proposals[:, 2], min=0, max=image_size[0])
                proposals[:, 3] = torch.clamp(proposals[:, 3], min=0, max=image_size[1])

                # 步骤 2: 按置信度得分降序排序，选取前 pre_nms_topk 个 proposals (取 min 防止 k 超过 anchor 总数)
                k_pre = min(self.pre_nms_topk, level_obj_logits[_batch_idx].shape[0])
                topk_scores, topk_indices = torch.topk(
                    level_obj_logits[_batch_idx], k=k_pre
                )
                proposals = proposals[topk_indices]

                # 步骤 3: 运行 NMS (非极大值抑制)，并保留前 post_nms_topk 个 proposals
                keep_indices = torchvision.ops.nms(proposals, topk_scores, self.nms_thresh)
                keep_indices = keep_indices[: self.post_nms_topk]
                level_proposals_per_image.append(proposals[keep_indices])
                ##############################################################
                #                        END OF YOUR CODE                    #
                ##############################################################

            # Collate proposals from individual images. Do not stack these
            # tensors, they may have different shapes since few images or
            # levels may have less than `post_nms_topk` proposals. We could
            # pad these tensors but there's no point - they will be used by
            # `torchvision.ops.roi_align` in second stage which operates
            # with lists, not batched tensors.
            proposals_all_levels[level_name] = level_proposals_per_image

        return proposals_all_levels

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)


class FasterRCNN(nn.Module):
    """
    Faster R-CNN detector: this module combines backbone, RPN, ROI predictors.

    Unlike Faster R-CNN, we will use class-agnostic box regression and Focal
    Loss for classification. We opted for this design choice for you to re-use
    a lot of concepts that you already implemented in FCOS - choosing one loss
    over other matters less overall.
    """

    def __init__(
        self,
        backbone: nn.Module,
        rpn: nn.Module,
        stem_channels: List[int],
        num_classes: int,
        batch_size_per_image: int,
        roi_size: Tuple[int, int] = (7, 7),
    ):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.batch_size_per_image = batch_size_per_image

        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules using `stem_channels` argument, exactly like
        # `FCOSPredictionNetwork` and `RPNPredictionNetwork`. use the same
        # stride, padding, and weight initialization as previous TODOs.
        #
        # HINT: This stem will be applied on RoI-aligned FPN features. You can
        # decide the number of input channels accordingly.
        ######################################################################
        # Fill this list. It is okay to use your implementation from
        # `FCOSPredictionNetwork` for this code block.
        cls_pred = []
        # Replace "pass" statement with your code
        cur_in_channels = backbone.out_channels
        for out_c in stem_channels:
            conv = nn.Conv2d(cur_in_channels, out_c, kernel_size=3,
                             stride=1, padding=1)
            nn.init.normal_(conv.weight, mean=0.0, std=0.01)
            nn.init.constant_(conv.bias, 0.0)

            cls_pred.append(conv)
            cls_pred.append(nn.ReLU())
            cur_in_channels = out_c
        ######################################################################
        # TODO: Add an `nn.Flatten` module to `cls_pred`, followed by a linear
        # layer to output C+1 classification logits (C classes + background).
        # Think about the input size of this linear layer based on the output
        # shape from `nn.Flatten` layer.
        ######################################################################
        # Replace "pass" statement with your code
        cls_pred.append(nn.Flatten())
        in_dense = stem_channels[-1] * self.roi_size[0] * self.roi_size[1]

        linear = nn.Linear(in_dense, num_classes + 1)
        nn.init.normal_(linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(linear.bias, 0.0)

        cls_pred.append(linear)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Wrap the layers defined by student into a `nn.Sequential` module,
        # Faster R-CNN also predicts box offsets to "refine" RPN proposals, we
        # exclude it for simplicity and keep RPN proposal boxes as final boxes.
        self.cls_pred = nn.Sequential(*cls_pred)

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        See documentation of `FCOS.forward` for more details.
        """

        feats_per_fpn_level = self.backbone(images)
        output_dict = self.rpn(
            feats_per_fpn_level, self.backbone.fpn_strides, gt_boxes
        )
        proposals_per_fpn_level = output_dict["proposals"]

        # Mix GT boxes with proposals. This is necessary to stabilize training
        # since RPN proposals may be bad during first few iterations. Also, why
        # waste good supervisory signal from GT boxes, for second-stage?
        if self.training:
            proposals_per_fpn_level = mix_gt_with_proposals(
                proposals_per_fpn_level, gt_boxes
            )

        # Get batch size from FPN feats:
        num_images = feats_per_fpn_level["p3"].shape[0]

        # Perform RoI-align using FPN features and proposal boxes.
        roi_feats_per_fpn_level = {
            level_name: None for level_name in feats_per_fpn_level.keys()
        }
        # Get RPN proposals from all levels.
        for level_name in feats_per_fpn_level.keys():
            ##################################################################
            # TODO: Call `torchvision.ops.roi_align`. See its documentation to
            # properly format input arguments. Use `aligned=True`
            ##################################################################
            level_feats = feats_per_fpn_level[level_name]
            level_props = output_dict["proposals"][level_name]
            level_stride = self.backbone.fpn_strides[level_name]

            # Replace "pass" statement with your code
            roi_feats = torchvision.ops.roi_align(
                level_feats,
                level_props,
                output_size=self.roi_size,
                spatial_scale=1.0 / level_stride,
                aligned=True,
            )
            ##################################################################
            #                         END OF YOUR CODE                       #
            ##################################################################

            roi_feats_per_fpn_level[level_name] = roi_feats

        # Combine ROI feats across FPN levels, do the same with proposals.
        # shape: (batch_size * total_proposals, fpn_channels, roi_h, roi_w)
        roi_feats = self._cat_across_fpn_levels(roi_feats_per_fpn_level, dim=0)

        # Obtain classification logits for all ROI features.
        # shape: (batch_size * total_proposals, num_classes)
        pred_cls_logits = self.cls_pred(roi_feats)

        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass. Batch size must be 1!
            # fmt: off
            return self.inference(
                images,
                proposals_per_fpn_level,
                pred_cls_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on

        ######################################################################
        # Match the RPN proposals with provided GT boxes and append to
        # `matched_gt_boxes`. Use `rcnn_match_anchors_to_gt` with IoU threshold
        # such that IoU > 0.5 is foreground, otherwise background.
        # There are no neutral proposals in second-stage.
        ######################################################################
        matched_gt_boxes = []
        for _idx in range(len(gt_boxes)):
            # Get proposals per image from this dictionary of list of tensors.
            proposals_per_fpn_level_per_image = {
                level_name: prop[_idx]
                for level_name, prop in output_dict["proposals"].items()
            }
            proposals_per_image = self._cat_across_fpn_levels(
                proposals_per_fpn_level_per_image, dim=0
            )
            gt_boxes_per_image = gt_boxes[_idx]
            # Replace "pass" statement with your code
            matched_gt_boxes.append(rcnn_match_anchors_to_gt(proposals_per_image, gt_boxes_per_image, iou_thresholds=(0.5, 0.5)))
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Combine predictions and GT from across all FPN levels.
        matched_gt_boxes = torch.cat(matched_gt_boxes, dim=0)

        ######################################################################
        # TODO: Train the classifier head. Perform these steps in order:
        #   1. Sample a few RPN proposals, like you sampled 50-50% anchor boxes
        #      to train RPN objectness classifier. However this time, sample
        #      such that ~25% RPN proposals are foreground, and the rest are
        #      background. Faster R-CNN performed such weighted sampling to
        #      deal with class imbalance, before Focal Loss was published.
        #
        #   2. Use these indices to get GT class labels from `matched_gt_boxes`
        #      and obtain the corresponding logits predicted by classifier.
        #
        #   3. Compute cross entropy loss - use `F.cross_entropy`, see its API
        #      documentation on PyTorch website. Since background ID = -1, you
        #      may shift class labels by +1 such that background ID = 0 and
        #      other VC classes have IDs (1-20). Make sure to reverse shift
        #      this during inference, so that model predicts VOC IDs (0-19).
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        loss_cls = None
        # Replace "pass" statement with your code
        total_batch_size = self.batch_size_per_image * num_images
        fg_idx, bg_idx = sample_rpn_training(
            matched_gt_boxes, total_batch_size, fg_fraction=0.25
        )

        sampled_indices = torch.cat([fg_idx, bg_idx])
        sampled_cls_logits = pred_cls_logits[sampled_indices]
        sampled_gt_classes = matched_gt_boxes[sampled_indices, 4].long()

        loss_cls = F.cross_entropy(sampled_cls_logits, sampled_gt_classes + 1)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return {
            "loss_rpn_obj": output_dict["loss_rpn_obj"],
            "loss_rpn_box": output_dict["loss_rpn_box"],
            "loss_cls": loss_cls,
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        proposals: torch.Tensor,
        pred_cls_logits: torch.Tensor,
        test_score_thresh: float,
        test_nms_thresh: float,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions.
        """

        # The second stage inference in Faster R-CNN is quite straightforward:
        # combine proposals from all FPN levels and perform a *class-specific
        # NMS*. There would have been more steps here if we further refined
        # RPN proposals by predicting box regression deltas.

        # Use `[0]` to remove the batch dimension.
        proposals = {level_name: prop[0] for level_name, prop in proposals.items()}
        pred_boxes = self._cat_across_fpn_levels(proposals, dim=0)

        ######################################################################
        # Faster R-CNN inference, perform the following steps in order:
        #   1. Get the most confident predicted class and score for every box.
        #      Note that the "score" of any class (including background) is its
        #      probability after applying C+1 softmax.
        #
        #   2. Only retain prediction that have a confidence score higher than
        #      provided threshold in arguments.
        #
        # NOTE: `pred_classes` may contain background as ID = 0 (based on how
        # the classifier was supervised in `forward`). Remember to shift the
        # predicted IDs such that model outputs ID (0-19) for 20 VOC classes.
        ######################################################################
        pred_scores, pred_classes = None, None
        # Replace "pass" statement with your code
        probs = F.softmax(pred_cls_logits, dim=-1)
        pred_scores, pred_classes = probs.max(dim=1)

        keep_mask = (pred_scores > test_score_thresh) & (pred_classes > 0)
        pred_boxes = pred_boxes[keep_mask]
        pred_scores = pred_scores[keep_mask]
        pred_classes = pred_classes[keep_mask] - 1
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        # STUDENTS: This line depends on your implementation of NMS.
        keep = class_spec_nms(
            pred_boxes, pred_scores, pred_classes, iou_threshold=test_nms_thresh
        )
        pred_boxes = pred_boxes[keep]
        pred_classes = pred_classes[keep]
        pred_scores = pred_scores[keep]
        return pred_boxes, pred_classes, pred_scores
