# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MaskRcnn based on ResNet50."""

import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore import context
from .resnet50 import ResNetFea, ResidualBlockUsing
from .bbox_assign_sample_stage2 import BboxAssignSampleForRcnn
from .fpn_neck import FeatPyramidNeck
from .proposal_generator import Proposal
from .rcnn_cls import RcnnCls
from .rcnn_mask import RcnnMask
from .rpn import RPN
from .roi_align import SingleRoIExtractor
from .anchor_generator import AnchorGenerator

class Mask_Rcnn_Resnet50(nn.Cell):
    """
    MaskRcnn Network.

    Note:
        backbone = resnet50

    Returns:
        Tuple, 输出张量的元组.
        rpn_loss: Scalar, RPN子网总损失.
        rcnn_loss: Scalar, RCNN子网的全部损失.
        rpn_cls_loss: Scalar, RPN子网分类损失.
        rpn_reg_loss: Scalar, RPN子网的回归损失.
        rcnn_cls_loss: Scalar, RCNNcls子网分类损失.
        rcnn_reg_loss: Scalar, RCNNcls子网的回归损失.
        rcnn_mask_loss: Scalar, RCNNmask子网掩码损失.

    Examples:
        net = Mask_Rcnn_Resnet50()
    """
    def __init__(self, config):
        # 这段代码初始化了 Mask_Rcnn_Resnet50 类，并基于提供的配置对象设置了几个实例变量
        super(Mask_Rcnn_Resnet50, self).__init__()

        if context.get_context("device_target") == "Ascend":
            self.cast_type = mstype.float16
            self.np_cast_type = np.float16
        else:
            self.cast_type = mstype.float32
            self.np_cast_type = np.float32

        self.train_batch_size = config.batch_size
        self.num_classes = config.num_classes
        self.anchor_scales = config.anchor_scales
        self.anchor_ratios = config.anchor_ratios
        self.anchor_strides = config.anchor_strides
        self.target_means = tuple(config.rcnn_target_means)
        self.target_stds = tuple(config.rcnn_target_stds)

        # 这段代码定义了一个目标检测中的锚点生成器，用于帮助目标检测模型预测图像中目标的位置。
        # Anchor generator
        anchor_base_sizes = None
        self.anchor_base_sizes = list(
            self.anchor_strides) if anchor_base_sizes is None else anchor_base_sizes

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, self.anchor_scales, self.anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)

        featmap_sizes = config.feature_shapes
        assert len(featmap_sizes) == len(self.anchor_generators)

        self.anchor_list = self.get_anchors(featmap_sizes)

        # Backbone resnet50：包含了 49 个卷积层、一个全连接层
        self.backbone = ResNetFea(ResidualBlockUsing,
                                  config.resnet_block,
                                  config.resnet_in_channels,
                                  config.resnet_out_channels,
                                  False)

        # Fpn：通过构造一种独特的特征金字塔来避免图像金字塔中计算量过高的问题，同时能够较好地处理目标检测中的多尺度变化问题
        self.fpn_ncek = FeatPyramidNeck(config.fpn_in_channels,
                                        config.fpn_out_channels,
                                        config.fpn_num_outs,
                                        config.feature_shapes)

        # Rpn and rpn loss：rpn专门用来提取候选框
        self.gt_labels_stage1 = Tensor(np.ones((self.train_batch_size, config.num_gts)).astype(np.uint8))
        self.rpn_with_loss = RPN(config,
                                 self.train_batch_size,
                                 config.rpn_in_channels,
                                 config.rpn_feat_channels,
                                 config.num_anchors,
                                 config.rpn_cls_out_channels)

        # Proposal：Proposal 类是一个提出区域的生成器，用于生成建议框
        self.proposal_generator = Proposal(config,
                                           self.train_batch_size,
                                           config.activate_num_classes,
                                           config.use_sigmoid_cls)
        self.proposal_generator.set_train_local(config, True)
        self.proposal_generator_test = Proposal(config,
                                                config.test_batch_size,
                                                config.activate_num_classes,
                                                config.use_sigmoid_cls)
        self.proposal_generator_test.set_train_local(config, False)

        # 这段代码定义了一个名为 self.bbox_assigner_sampler_for_rcnn 的类，并使用 BboxAssignSampleForRcnn 类初始化了该实例。
        # 该类被用于为第二阶段的 RCNN 网络分配和采样边界框。
        # Assign and sampler stage two
        self.bbox_assigner_sampler_for_rcnn = BboxAssignSampleForRcnn(config, self.train_batch_size,
                                                                      config.num_bboxes_stage2, True)
        self.decode = P.BoundingBoxDecode(max_shape=(768, 1280), means=self.target_means, \
                                          stds=self.target_stds)

        # 调用池化层
        # Roi
        self.init_roi(config)

        # 这段代码定义了 rcnn_cls 和 rcnn_mask 两个子网络，并使用 RcnnCls 和 RcnnMask 类初始化了这两个子网络。
        # Rcnn
        self.rcnn_cls = RcnnCls(config, self.train_batch_size, self.num_classes)
        self.rcnn_mask = RcnnMask(config, self.train_batch_size, self.num_classes)

        # Op declare
        self.squeeze = P.Squeeze() # 删除维度为1的维度
        self.cast = P.Cast() # 数据类型转换

        self.concat = P.Concat(axis=0) # 沿着不同的轴连接多个张量
        self.concat_1 = P.Concat(axis=1)
        self.concat_2 = P.Concat(axis=2)

        self.reshape = P.Reshape() # 改变张量形状
        self.select = P.Select() # 选择张量
        self.greater = P.Greater() # 比较张量大小
        self.transpose = P.Transpose() # 转置张量

        # Test mode
        self.init_test_mode(config)

        # 用于控制在分类过程中拼接张量的范围，以减少计算量和内存使用。
        # Improve speed
        self.concat_start = min(self.num_classes - 2, 55)
        self.concat_end = (self.num_classes - 1)

        # Init tensor
        self.init_tensor(config)

    def init_roi(self, config):
        """"
        提取
        """
        # 提取ROI特征向量
        self.roi_align = SingleRoIExtractor(config,
                                            config.roi_layer,
                                            config.roi_align_out_channels,
                                            config.roi_align_featmap_strides,
                                            self.train_batch_size,
                                            config.roi_align_finest_scale,
                                            mask=False)
        self.roi_align.set_train_local(config, True)

        # 提取掩码特征向量
        self.roi_align_mask = SingleRoIExtractor(config,
                                                 config.roi_layer,
                                                 config.roi_align_out_channels,
                                                 config.roi_align_featmap_strides,
                                                 self.train_batch_size,
                                                 config.roi_align_finest_scale,
                                                 mask=True)
        self.roi_align_mask.set_train_local(config, True)
        
        # 测试
        self.roi_align_test = SingleRoIExtractor(config,
                                                 config.roi_layer,
                                                 config.roi_align_out_channels,
                                                 config.roi_align_featmap_strides,
                                                 1,
                                                 config.roi_align_finest_scale,
                                                 mask=False)
        self.roi_align_test.set_train_local(config, False)

        self.roi_align_mask_test = SingleRoIExtractor(config,
                                                      config.roi_layer,
                                                      config.roi_align_out_channels,
                                                      config.roi_align_featmap_strides,
                                                      1,
                                                      config.roi_align_finest_scale,
                                                      mask=True)
        self.roi_align_mask_test.set_train_local(config, False)

    def init_test_mode(self, config):
        """
        初始化测试模式下的一些张量和操作对象
        """
        self.test_batch_size = config.test_batch_size
        self.split = P.Split(axis=0, output_num=self.test_batch_size) # 将张量沿着指定轴分割为多个子张量
        self.split_shape = P.Split(axis=0, output_num=4) 
        self.split_scores = P.Split(axis=1, output_num=self.num_classes)
        self.split_fb_mask = P.Split(axis=1, output_num=self.num_classes)
        self.split_cls = P.Split(axis=0, output_num=self.num_classes-1)
        self.tile = P.Tile() # 用于在指定轴上重复张量多次
        self.gather = P.GatherNd() # 指定索引处获取张量数据

        self.rpn_max_num = config.rpn_max_num # 检测中使用的最大RPN（Region Proposal Network）候选框数量。

        self.zeros_for_nms = Tensor(np.zeros((self.rpn_max_num, 3)).astype(self.np_cast_type))
        self.ones_mask = np.ones((self.rpn_max_num, 1)).astype(np.bool)
        self.zeros_mask = np.zeros((self.rpn_max_num, 1)).astype(np.bool)
        self.bbox_mask = Tensor(np.concatenate((self.ones_mask, self.zeros_mask,
                                                self.ones_mask, self.zeros_mask), axis=1))
        self.nms_pad_mask = Tensor(np.concatenate((self.ones_mask, self.ones_mask,
                                                   self.ones_mask, self.ones_mask, self.zeros_mask), axis=1))

        self.test_score_thresh = Tensor(np.ones((self.rpn_max_num, 1)).astype(self.np_cast_type) * \
                                        config.test_score_thr)
        self.test_score_zeros = Tensor(np.ones((self.rpn_max_num, 1)).astype(self.np_cast_type) * 0)
        self.test_box_zeros = Tensor(np.ones((self.rpn_max_num, 4)).astype(self.np_cast_type) * -1)
        self.test_iou_thr = Tensor(np.ones((self.rpn_max_num, 1)).astype(self.np_cast_type) * config.test_iou_thr)
        self.test_max_per_img = config.test_max_per_img
        self.nms_test = P.NMSWithMask(config.test_iou_thr)
        self.softmax = P.Softmax(axis=1)
        self.logicand = P.LogicalAnd()
        self.oneslike = P.OnesLike()
        self.test_topk = P.TopK(sorted=True)
        self.test_num_proposal = self.test_batch_size * self.rpn_max_num

    def init_tensor(self, config):
        """
        初始化一些张量和操作对象
        """
        roi_align_index = [np.array(np.ones((config.num_expected_pos_stage2 + \
                                             config.num_expected_neg_stage2, 1)) * i,
                                    dtype=self.np_cast_type) for i in range(self.train_batch_size)]

        roi_align_index_test = [np.array(np.ones((config.rpn_max_num, 1)) * i, dtype=self.np_cast_type) \
                                for i in range(self.test_batch_size)]

        self.roi_align_index_tensor = Tensor(np.concatenate(roi_align_index))
        self.roi_align_index_test_tensor = Tensor(np.concatenate(roi_align_index_test))

        roi_align_index_pos = [np.array(np.ones((config.num_expected_pos_stage2, 1)) * i,
                                        dtype=self.np_cast_type) for i in range(self.train_batch_size)]
        self.roi_align_index_tensor_pos = Tensor(np.concatenate(roi_align_index_pos))

        self.rcnn_loss_cls_weight = Tensor(np.array(config.rcnn_loss_cls_weight).astype(self.np_cast_type))
        self.rcnn_loss_reg_weight = Tensor(np.array(config.rcnn_loss_reg_weight).astype(self.np_cast_type))
        self.rcnn_loss_mask_fb_weight = Tensor(np.array(config.rcnn_loss_mask_fb_weight).astype(self.np_cast_type))

        self.argmax_with_value = P.ArgMaxWithValue(axis=1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.onehot = P.OneHot()
        self.reducesum = P.ReduceSum()
        self.sigmoid = P.Sigmoid()
        self.expand_dims = P.ExpandDims()
        self.test_mask_fb_zeros = Tensor(np.zeros((self.rpn_max_num, 28, 28)).astype(self.np_cast_type))
        self.value = Tensor(1.0, self.cast_type)

    def construct(self, img_data, img_metas, gt_bboxes, gt_labels, gt_valids, gt_masks):
        """
        构造 Mask R-CNN 网络.
        接受输入的图像数据img_data、图像元数据img_metas、真实边界框gt_bboxes、真实标签gt_labels、
        真实掩码gt_masks等,并根据这些数据计算出网络的输出和损失。

        """

        # 将输入的图像数据通过self.backbone和self.fpn_ncek方法分别进行特征提取和FPN操作，得到特征图x
        x = self.backbone(img_data)
        x = self.fpn_ncek(x)

        # 调用self.rpn_with_loss方法计算RPN网络的输出和损失，包括RPN分类得分、边界框预测值、RPN分类损失、RPN回归损失等
        rpn_loss, cls_score, bbox_pred, rpn_cls_loss, rpn_reg_loss, _ = self.rpn_with_loss(x,
                                                                                           img_metas,
                                                                                           self.anchor_list,
                                                                                           gt_bboxes,
                                                                                           self.gt_labels_stage1,
                                                                                           gt_valids)

        # 调用self.proposal_generator方法或self.proposal_generator_test方法生成候选框和掩码
        if self.training:
            proposal, proposal_mask = self.proposal_generator(cls_score, bbox_pred, self.anchor_list)
        else:
            proposal, proposal_mask = self.proposal_generator_test(cls_score, bbox_pred, self.anchor_list)

        gt_labels = self.cast(gt_labels, mstype.int32)
        gt_valids = self.cast(gt_valids, mstype.int32)
        bboxes_tuple = ()
        deltas_tuple = ()
        labels_tuple = ()
        mask_tuple = ()

        pos_bboxes_tuple = ()
        pos_mask_fb_tuple = ()
        pos_labels_tuple = ()
        pos_mask_tuple = ()

        # 在训练模式下，根据真实边界框和生成的候选框，计算RCNN网络的输入和标签
        if self.training:
            for i in range(self.train_batch_size):
                gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])

                gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
                gt_labels_i = self.cast(gt_labels_i, mstype.uint8)

                gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])
                gt_valids_i = self.cast(gt_valids_i, mstype.bool_)

                gt_masks_i = self.squeeze(gt_masks[i:i + 1:1, ::])
                gt_masks_i = self.cast(gt_masks_i, mstype.bool_)

                # 调用self.bbox_assigner_sampler_for_rcnn方法计算真实边界框的偏移量、标签和掩码等
                bboxes, deltas, labels, mask, pos_bboxes, pos_mask_fb, pos_labels, pos_mask = \
                    self.bbox_assigner_sampler_for_rcnn(gt_bboxes_i,
                                                        gt_labels_i,
                                                        proposal_mask[i],
                                                        proposal[i][::, 0:4:1],
                                                        gt_valids_i,
                                                        gt_masks_i)
                bboxes_tuple += (bboxes,)
                deltas_tuple += (deltas,)
                labels_tuple += (labels,)
                mask_tuple += (mask,)

                pos_bboxes_tuple += (pos_bboxes,)
                pos_mask_fb_tuple += (pos_mask_fb,)
                pos_labels_tuple += (pos_labels,)
                pos_mask_tuple += (pos_mask,)

            bbox_targets = self.concat(deltas_tuple)
            rcnn_labels = self.concat(labels_tuple)
            bbox_targets = F.stop_gradient(bbox_targets)
            rcnn_labels = F.stop_gradient(rcnn_labels)
            rcnn_labels = self.cast(rcnn_labels, mstype.int32)

            rcnn_pos_masks_fb = self.concat(pos_mask_fb_tuple)
            rcnn_pos_masks_fb = F.stop_gradient(rcnn_pos_masks_fb)
            rcnn_pos_labels = self.concat(pos_labels_tuple)
            rcnn_pos_labels = F.stop_gradient(rcnn_pos_labels)
            rcnn_pos_labels = self.cast(rcnn_pos_labels, mstype.int32)
        # 在测试模式下，由于没有真实边界框，所以直接使用生成的候选框进行后续操作
        else:
            mask_tuple += proposal_mask
            bbox_targets = proposal_mask
            rcnn_labels = proposal_mask

            rcnn_pos_masks_fb = proposal_mask
            rcnn_pos_labels = proposal_mask
            for p_i in proposal:
                bboxes_tuple += (p_i[::, 0:4:1],)

        bboxes_all, rois, pos_rois = self.rois(bboxes_tuple, pos_bboxes_tuple)

        # 在训练模式下，根据ROI特征和真实边界框的偏移量、标签和掩码等，计算RCNN网络的分类和回归损失
        if self.training:
            roi_feats = self.roi_align(rois,
                                       self.cast(x[0], mstype.float32),
                                       self.cast(x[1], mstype.float32),
                                       self.cast(x[2], mstype.float32),
                                       self.cast(x[3], mstype.float32))
        # 在测试模式下，由于没有真实边界框，所以直接使用生成的候选框进行ROI特征提取和分类回归
        else:
            roi_feats = self.roi_align_test(rois,
                                            self.cast(x[0], mstype.float32),
                                            self.cast(x[1], mstype.float32),
                                            self.cast(x[2], mstype.float32),
                                            self.cast(x[3], mstype.float32))


        roi_feats = self.cast(roi_feats, self.cast_type)
        rcnn_masks = self.concat(mask_tuple)
        rcnn_masks = F.stop_gradient(rcnn_masks)
        rcnn_mask_squeeze = self.squeeze(self.cast(rcnn_masks, mstype.bool_))

        rcnn_pos_masks = self.concat(pos_mask_tuple)
        rcnn_pos_masks = F.stop_gradient(rcnn_pos_masks)
        rcnn_pos_mask_squeeze = self.squeeze(self.cast(rcnn_pos_masks, mstype.bool_))

        rcnn_cls_loss, rcnn_reg_loss = self.rcnn_cls(roi_feats,
                                                     bbox_targets,
                                                     rcnn_labels,
                                                     rcnn_mask_squeeze)

        if self.training:
            return self.get_output_train(pos_rois, x, rcnn_pos_labels, rcnn_pos_mask_squeeze, rcnn_pos_masks_fb,
                                         rpn_loss, rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss)

        return self.get_output_eval(x, bboxes_all, rcnn_cls_loss, rcnn_reg_loss, rcnn_masks, img_metas)

    def rois(self, bboxes_tuple, pos_bboxes_tuple):
        """
        用于生成 ROIs(Region of Interests)特征
        """
        pos_rois = None
        if self.training:
        # 如果当前批次中有多个样本，则将所有的边界框和正样本的边界框连接成一个张量，
        # 否则直接使用第一个样本的边界框和正样本的边界框
            if self.train_batch_size > 1:
                bboxes_all = self.concat(bboxes_tuple)
                pos_bboxes_all = self.concat(pos_bboxes_tuple)
            else:
                bboxes_all = bboxes_tuple[0]
                pos_bboxes_all = pos_bboxes_tuple[0]
            rois = self.concat_1((self.roi_align_index_tensor, bboxes_all)) # 调用self.concat_1方法将ROI索引张量和所有的边界框连接成一个张量
            pos_rois = self.concat_1((self.roi_align_index_tensor_pos, pos_bboxes_all))
            pos_rois = self.cast(pos_rois, mstype.float32) # 并对其进行类型转换
            pos_rois = F.stop_gradient(pos_rois) # 同时使用F.stop_gradient方法防止梯度回传。
        else:
        # 如果当前网络处于测试模式，直接将所有的边界框连接成一个张量，并调用self.roi_align_index_test_tensor方法生成ROI索引张量
            if self.test_batch_size > 1:
                bboxes_all = self.concat(bboxes_tuple)
            else:
                bboxes_all = bboxes_tuple[0]
            rois = self.concat_1((self.roi_align_index_test_tensor, bboxes_all))

        # 将生成的边界框的数据类型转换为float32类型，并使用F.stop_gradient方法防止梯度回传
        rois = self.cast(rois, mstype.float32)
        rois = F.stop_gradient(rois)

        # bboxes_all表示所有的边界框坐标，rois表示所有的ROI特征，pos_rois表示所有正样本的ROI特征
        return bboxes_all, rois, pos_rois

    def get_output_train(self, pos_rois, x, rcnn_pos_labels, rcnn_pos_mask_squeeze, rcnn_pos_masks_fb,
                         rpn_loss, rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss):
        """
        用于在训练阶段计算网络的输出结果
        """
        output = ()
       # 根据正样本的ROI特征和特征图，调用self.roi_align_mask方法生成ROI特征掩码
        roi_feats_mask = self.roi_align_mask(pos_rois,
                                             self.cast(x[0], mstype.float32),
                                             self.cast(x[1], mstype.float32),
                                             self.cast(x[2], mstype.float32),
                                             self.cast(x[3], mstype.float32))
        # 将ROI特征掩码类型转换为self.cast_type类型，其中self.cast_type是初始化RCNN网络时指定的数据类型
        roi_feats_mask = self.cast(roi_feats_mask, self.cast_type)
        # 根据ROI特征掩码、正样本的标签、掩码和边界框，调用self.rcnn_mask方法计算掩码损失
        rcnn_mask_fb_loss = self.rcnn_mask(roi_feats_mask,
                                           rcnn_pos_labels,
                                           rcnn_pos_mask_squeeze,
                                           rcnn_pos_masks_fb)

        # 根据RPN网络的分类和回归损失、RCNN网络的分类和回归损失以及掩码损失，计算RCNN网络的总损失
        rcnn_loss = self.rcnn_loss_cls_weight * rcnn_cls_loss + self.rcnn_loss_reg_weight * rcnn_reg_loss + \
                    self.rcnn_loss_mask_fb_weight * rcnn_mask_fb_loss
        output += (rpn_loss, rcnn_loss, rpn_cls_loss, rpn_reg_loss,
                   rcnn_cls_loss, rcnn_reg_loss, rcnn_mask_fb_loss)
        return output

    def get_output_eval(self, x, bboxes_all, rcnn_cls_loss, rcnn_reg_loss, rcnn_masks, img_metas):
        """
        用于在推理阶段计算网络的输出结果
        """
        # 根据特征图和所有的边界框坐标，调用self.rcnn_mask_test方法生成掩码预测
        mask_fb_pred_all = self.rcnn_mask_test(x, bboxes_all, rcnn_cls_loss, rcnn_reg_loss)
        # 根据掩码预测、所有的边界框坐标、RCNN网络的分类和回归损失以及图像元数据，调用self.get_det_bboxes方法生成检测结果
        output = self.get_det_bboxes(rcnn_cls_loss, rcnn_reg_loss, rcnn_masks, bboxes_all,
                                     img_metas, mask_fb_pred_all)
        return output

    def get_det_bboxes(self, cls_logits, reg_logits, mask_logits, rois, img_metas, mask_fb_pred_all):
        """
        获取实际的检测盒
        """
        scores = self.softmax(cls_logits / self.value) # 得到每个类别的概率得分
        mask_fb_logits = self.sigmoid(mask_fb_pred_all) # 得到每个像素点是否属于目标的概率

        boxes_all = ()
        for i in range(self.num_classes):
            k = i * 4
            reg_logits_i = self.squeeze(reg_logits[::, k:k+4:1])
            out_boxes_i = self.decode(rois, reg_logits_i) # 调用self.decode方法解码出每个类别的边界框坐标
            boxes_all += (out_boxes_i,)

        img_metas_all = self.split(img_metas) # 拆分成每张图像的结果
        scores_all = self.split(scores)
        mask_all = self.split(self.cast(mask_logits, mstype.int32))
        mask_fb_all = self.split(mask_fb_logits)

        boxes_all_with_batchsize = ()
        for i in range(self.test_batch_size):
            scale = self.split_shape(self.squeeze(img_metas_all[i]))
            scale_h = scale[2]
            scale_w = scale[3]
            boxes_tuple = ()
            for j in range(self.num_classes):
                boxes_tmp = self.split(boxes_all[j])
                out_boxes_h = boxes_tmp[i] / scale_h
                out_boxes_w = boxes_tmp[i] / scale_w
                # 根据掩码预测结果生成掩码，同时将当前类别在当前图像中的边界框坐标和掩码添加到boxes_tuple中
                boxes_tuple += (self.select(self.bbox_mask, out_boxes_w, out_boxes_h),)
            boxes_all_with_batchsize += (boxes_tuple,)
        # 根据预测的边界框坐标和得分，使用多类别非极大值抑制（multiclass NMS）方法生成最终的检测结果
        output = self.multiclass_nms(boxes_all_with_batchsize, scores_all, mask_all, mask_fb_all)

        return output

    def multiclass_nms(self, boxes_all, scores_all, mask_all, mask_fb_all):
        """
        多尺度后处理.
        实现了多类别非极大值抑制(multiclass NMS)算法的后处理部分,用于对检测结果进行过滤和筛选,输出最终的检测结果
        """
        all_bboxes = ()
        all_labels = ()
        all_masks = ()
        all_masks_fb = ()

        for i in range(self.test_batch_size):
            bboxes = boxes_all[i]
            scores = scores_all[i]
            masks = self.cast(mask_all[i], mstype.bool_)
            masks_fb = mask_fb_all[i]
            _mask_fb_all = self.split_fb_mask(masks_fb)

            res_boxes_tuple = ()
            res_labels_tuple = ()
            res_masks_tuple = ()
            res_masks_fb_tuple = ()

            for j in range(self.num_classes - 1):
                k = j + 1
                _cls_scores = scores[::, k:k + 1:1]
                _bboxes = self.squeeze(bboxes[k])
                _mask_o = self.reshape(masks, (self.rpn_max_num, 1))
                _masks_fb = self.squeeze(_mask_fb_all[k])

                cls_mask = self.greater(_cls_scores, self.test_score_thresh)
                _mask = self.logicand(_mask_o, cls_mask)

                _reg_mask = self.cast(self.tile(self.cast(_mask, mstype.int32), (1, 4)), mstype.bool_)

                _bboxes = self.select(_reg_mask, _bboxes, self.test_box_zeros)
                _fb_mask = self.expand_dims(_mask, -1)
                _mask_fb_mask = self.cast(self.tile(self.cast(_fb_mask, mstype.int32), (1, 28, 28)), mstype.bool_)
                _masks_fb = self.select(_mask_fb_mask, _masks_fb, self.test_mask_fb_zeros)
                _cls_scores = self.select(_mask, _cls_scores, self.test_score_zeros)
                __cls_scores = self.squeeze(_cls_scores)
                scores_sorted, topk_inds = self.test_topk(__cls_scores, self.rpn_max_num)
                topk_inds = self.reshape(topk_inds, (self.rpn_max_num, 1))
                scores_sorted = self.reshape(scores_sorted, (self.rpn_max_num, 1))
                _bboxes_sorted = self.gather(_bboxes, topk_inds)
                _mask_fb_sorted = self.gather(_masks_fb, topk_inds)
                _mask_sorted = self.gather(_mask, topk_inds)

                scores_sorted = self.tile(scores_sorted, (1, 4))
                cls_dets = self.concat_1((_bboxes_sorted, scores_sorted))
                cls_dets = P.Slice()(cls_dets, (0, 0), (self.rpn_max_num, 5))

                cls_dets, _index, _mask_nms = self.nms_test(cls_dets)
                _index = self.reshape(_index, (self.rpn_max_num, 1))
                _mask_nms = self.reshape(_mask_nms, (self.rpn_max_num, 1))

                _mask_n = self.gather(_mask_sorted, _index)
                _mask_n = self.logicand(_mask_n, _mask_nms)

                _mask_fb = self.gather(_mask_fb_sorted, _index)

                cls_labels = self.oneslike(_index) * j
                res_boxes_tuple += (cls_dets,)
                res_labels_tuple += (cls_labels,)
                res_masks_tuple += (_mask_n,)
                res_masks_fb_tuple += (_mask_fb,)

            res_boxes_start = self.concat(res_boxes_tuple[:self.concat_start])
            res_labels_start = self.concat(res_labels_tuple[:self.concat_start])
            res_masks_start = self.concat(res_masks_tuple[:self.concat_start])
            res_masks_fb_start = self.concat(res_masks_fb_tuple[:self.concat_start])

            res_boxes_end = self.concat(res_boxes_tuple[self.concat_start:self.concat_end])
            res_labels_end = self.concat(res_labels_tuple[self.concat_start:self.concat_end])
            res_masks_end = self.concat(res_masks_tuple[self.concat_start:self.concat_end])
            res_masks_fb_end = self.concat(res_masks_fb_tuple[self.concat_start:self.concat_end])

            res_boxes = self.concat((res_boxes_start, res_boxes_end))
            res_labels = self.concat((res_labels_start, res_labels_end))
            res_masks = self.concat((res_masks_start, res_masks_end))
            res_masks_fb = self.concat((res_masks_fb_start, res_masks_fb_end))

            reshape_size = (self.num_classes - 1) * self.rpn_max_num
            res_boxes = self.reshape(res_boxes, (1, reshape_size, 5))
            res_labels = self.reshape(res_labels, (1, reshape_size, 1))
            res_masks = self.reshape(res_masks, (1, reshape_size, 1))
            res_masks_fb = self.reshape(res_masks_fb, (1, reshape_size, 28, 28))

            all_bboxes += (res_boxes,)
            all_labels += (res_labels,)
            all_masks += (res_masks,)
            all_masks_fb += (res_masks_fb,)

        all_bboxes = self.concat(all_bboxes)
        all_labels = self.concat(all_labels)
        all_masks = self.concat(all_masks)
        all_masks_fb = self.concat(all_masks_fb)
        return all_bboxes, all_labels, all_masks, all_masks_fb

    def get_anchors(self, featmap_sizes):
        """
        根据特征图的大小获取锚

        Args:
            featmap_sizes (list[tuple]): 多层次特征图大小.
            img_metas (list[dict]): 图像元信息.

        Returns:
            tuple: 每个图像的锚点，每个图像的有效标志
        """
        num_levels = len(featmap_sizes)

        # 由于所有图像的特征图大小相同，因此我们只计算一次锚点
        multi_level_anchors = ()
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors += (Tensor(anchors.astype(self.np_cast_type)),)

        return multi_level_anchors

    def rcnn_mask_test(self, x, rois, cls_pred, reg_pred):
        """
        Prediction masks in an images by the bounding boxes
        通过边界框对图像的mask进行预测
        """
        cls_scores = self.softmax(cls_pred / self.value)

        cls_scores_all = self.split(cls_scores)
        reg_pred = self.reshape(reg_pred, (-1, self.num_classes, 4))
        reg_pred_all = self.split(reg_pred)
        rois_all = self.split(rois)
        boxes_tuple = ()
        for i in range(self.test_batch_size):
            cls_score_max_index, _ = self.argmax_with_value(cls_scores_all[i])
            cls_score_max_index = self.cast(self.onehot(cls_score_max_index, self.num_classes,
                                                        self.on_value, self.off_value), self.cast_type)
            cls_score_max_index = self.expand_dims(cls_score_max_index, -1)
            cls_score_max_index = self.tile(cls_score_max_index, (1, 1, 4))
            reg_pred_max = reg_pred_all[i] * cls_score_max_index
            reg_pred_max = self.reducesum(reg_pred_max, 1)
            out_boxes_i = self.decode(rois_all[i], reg_pred_max)
            boxes_tuple += (out_boxes_i,)

        boxes_all = self.concat(boxes_tuple)
        boxes_rois = self.concat_1((self.roi_align_index_test_tensor, boxes_all))
        boxes_rois = self.cast(boxes_rois, self.cast_type)
        roi_feats_mask_test = self.roi_align_mask_test(boxes_rois,
                                                       self.cast(x[0], mstype.float32),
                                                       self.cast(x[1], mstype.float32),
                                                       self.cast(x[2], mstype.float32),
                                                       self.cast(x[3], mstype.float32))
        roi_feats_mask_test = self.cast(roi_feats_mask_test, self.cast_type)
        mask_fb_pred_all = self.rcnn_mask(roi_feats_mask_test)
        return mask_fb_pred_all

# 封装了一个用于推理的Mask R-CNN模型，用于对输入图像进行目标检测和实例分割
class MaskRcnn_Infer(nn.Cell):
    def __init__(self, config):
        super(MaskRcnn_Infer, self).__init__()
        self.network = Mask_Rcnn_Resnet50(config)
        self.network.set_train(False)

    def construct(self, img_data, img_metas):
        """
        用于执行模型的前向传播，得到输出结果
        """
        output = self.network(img_data, img_metas, None, None, None, None)
        return output
