import torch.nn as nn
import torch
from torchvision.ops.boxes import nms as nms_torch
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from efficientnet import EfficientNet as EffNet
from efficientnet.utils import MemoryEfficientSwish, Swish
from efficientnet.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding
from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
INF = 1e8

def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


class DetHead(nn.Module):
    
    def __init__(self, num_classes, in_channels, feat_channels=88, stacked_convs=4, 
                    strides=(4, 8, 16, 32, 64)):
        super(DetHead, self).__init__()
        self.num_classes = num_classes
        # self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = ((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF))
        self.norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.center_sampling = True
        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs): # 4 * convs
            c = self.in_channels if i == 0 else self.feat_channels # d1: 88
            # classification convolutional layers
            # waiting add....
            self.cls_convs.append(
                ConvModule(
                    c,                  # input channels
                    self.feat_channels, # output channels KEEP SAME INPUT & OUTPUT SIZE
                    3,                  # 3 x 3 kernel size
                    stride = 1, 
                    padding = 1,
                    norm_cfg = self.norm_cfg,
                    bias = self.norm_cfg is None 
                )
            )

            self.reg_convs.append(
                ConvModule(
                    c,
                    self.feat_channels,
                    3,
                    stride = 1, 
                    padding = 1, 
                    normal_cfg = self.norm_cfg,
                    bias = self.norm_cfg is None
                )
            )
        
        self.fcos_cls = nn.Conv2d(self.feat_channels, self.num_classes, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1) # *l, *r, *t, *b
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1) # criterion
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides]) # [4, 8, 16, 32, 64]

    def forward(self, feats):
        return multi_apply(self.forward_single_scale, feats, self.scales)
        # cls_scores, bbox_preds, centernesses

    def forward_single_scale(self, x, scale):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(x)
            # 2 branches: clssification & centerness
            cls_score = self.fcos_cls(cls_feat)
            centerness = self.fcos_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_layer)
            # regression elems: *l, *r, *t, *b
            bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp() # scale the bbox_pred of different level
            # (0, +oo)
        return cls_score, bbox_pred, centerness

    def loss(self, cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels):
        """Calculate loss of boxNet
        Args:
            cls_scores, bbox_preds, centernesses: list A Series of ...
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses), \
            "mismatching of cls_scores, bbox_preds, centernesses."
        featmap_sizes = [featmap.shape for featmap in cls_scores] # default equal 
        # all points on feature maps | meshgrid of axes point of different scales
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        # 
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes, gt_labels)


    def get_targets(self, points, gt_bboxes_one_batch, gt_labels_one_batch):
        assert len(points) == len(self.regress_ranges), \
            "make sure if the same nums of scales ranges" 
        num_levels = len(points) # different levels(scales)
        # expand regress ranges as the points
        expanded_regress_ranges = [   # elem               # shape
            points[i].new_tensor(self.regress_ranges)[None].expand_as(
                points[i]) for i in range(len(points)) # ←ranges shape: [num_levels, 2]
        ]
        # concatenate points and ranges 
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim = 0)
        concat_points = torch.cat(points, dim = 0)
        # Get one batch results(labels + targets of lrtb)
        labels_list, bbox_targets_list = multi_apply(
            self.target_single, gt_bboxes_one_batch, gt_labels_one_batch,
            concat_points, concat_regress_ranges
        )
        # points in featuer maps are centres of affined bbox 代表每个level里anchor点的数目
        num_anchorPoints_perLevel = [center.size(0) for center in points] 
        # 根据每个level里anchor点的数目拆分每张图的label
        # labels_list每个元素是当前图的每个level里的每个点的标签
        labels_list = [label_list.split(num_anchorPoints_perLevel, 0) \
                                        for label_list in labels_list]
        

        


    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        """ Each img processing
        Args: 
            gt_bboxes, gt_labels: ground truth 
            points: points in feature maps
            regress_ranges: confine the range of positive points 
        Return:
            labels & lrtb of ONE img
        Modified by: Daivd Qiao
        """
        num_points = points.size(0) # diff from GET_TARGETS: num_levels - len(points)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points, ), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * ( # gt_bboxes shape: [num_gts, 4]
            gt_bboxes[:, 3] - gt_bboxes[:, 1]) # ←areas shape: [num_gts, ]
        
        areas = areas[None].repeat(num_points, 1) # ← areas shape: [num_points, num_gts]
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2) 
        # ↑ input ranges shape: [num_points, 2] | ↑ ranges shape: [num_points, num_gts, 4]
        # gt_bboxes input shape: [num_gts, 4]
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)

        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts) # shape: [num_points，num_gts]
        # 每个点和每个框之间的上下左右的差值
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        # bbox_targets ↓ shape: [num_points, num_gts，4]
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # Conditional Convolutions for Instance Segmentation
            # https://arxiv.org/abs/2003.05664
            # If the mapped location falls in thecenter region of an instance, 
            # the location is considered to beresponsible for the instance. 
            # Any locations outside the cen-ter regions are labeled as negative samples. 
            # The center re-gion is defined as the box (cx−rs,cy−rs,cx+rs,cy+rs)
            # 's' is the down-sampling ratio of Pi and 'r' is a constant scalar 
            radius = self.center_sample_radius
            




        # 找到（l,r,t,b）中最小的，如果最小的大于０，那么这个点肯定在对应的gt框里面
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        areas[inside_gt_bbox_mask == 0] = INF # 将框外面的点对应的area置为无穷

        # 找到（l,r,t,b）中最大的，如果最大的满足范围约束
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])
        areas[inside_regress_range == 0] = INF # 将不满足范围约束的也置为无穷
        # 找到每个点对应的面积最小的gt框
        min_area, min_area_inds = areas.min(dim = 1) # min_area shape: [num_points, ]
        # ↓labels shape: [num_gts, ] 
        labels = gt_labels[min_area_inds] # shape: [num_points, ]
        # areas[inside_gt_bbox_mask == 0] = INF # 将框外面的点对应的area置为无穷
        labels[min_area == INF] = 0 # 把负样本置０
        bbox_targets = bbox_targets[range(num_points), min_area_inds] # 正样本提取 shape: [num_points, 4]

        return labels, bbox_targets # labels 和 bbox_targets 是配套的

    def get_points(self, featmap_sizes, dtype, device):
        """ Get points as to feature map sizes with original axes
        Args:
            featmap_sizes: MULTIPLE-level feature map sizes
        """
        multipleLevel_points = []
        for i in range(len(featmap_sizes)):
            multipleLevel_points.append(self._get_points_single(featmap_sizes[i], 
                                            self.strides[i], dtype, device))
        return multipleLevel_points

    def _get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_axes_arange = torch.arange(0, w * stride, stride, dtype=dtype, device=device)
        y_axes_arange = torch.arange(0, h * stride, stride, dtype=dtype, device=device)
        
        x, y = torch.meshgrid(x_axes_arange, y_axes_arange)
        # a location(x,y)on thefeature maps can be mapped back onto the input image as(『s/2』 + xs, 『s/2』 + ys)
        axes_points = torch.stack((x.reshape(-1), y.reshape(-1)), dim = 1) + stride // 2 # start point
        # If the mapped location falls in thecenter region of an instance, 
        # the location is considered to beresponsible for the instance. 
        return axes_points

class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPN(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class Regressor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])
        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats


class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()

        return feats


class EfficientNet(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, compound_coef, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}', load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[1:]


if __name__ == '__main__':
    from tensorboardX import SummaryWriter


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
