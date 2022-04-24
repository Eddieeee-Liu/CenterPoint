import torch
import torch.nn as nn

#import MinkowskiEngine as ME
import spconv.pytorch as spconv

from functools import partial
from spconv.core import ConvAlgo
import copy
import time


class DynamicPruningConv(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, bias=False, pruning_ratio=0.5):
        super().__init__()
        self.pruning_ratio = pruning_ratio
        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels


        self.pred_conv = spconv.SubMConv3d(
                in_channels,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=int(kernel_size//2),
                bias=False,
                indice_key=indice_key + "_pred_conv",
                algo=ConvAlgo.Native
            )

        self.conv_block = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=bias,
                                        indice_key=indice_key,
                                        # subm_torch=False,
                                        algo=ConvAlgo.Native
                                    )

        # self.channel_align = spconv.SubMConv3d(
        #                                 in_channels,
        #                                 out_channels,
        #                                 kernel_size=1,
        #                                 stride=1,
        #                                 bias=bias,
        #                                 padding=0,
        #                                 indice_key=indice_key + "_channel_align",
        #                                 # subm_torch=False,
        #                                 algo=ConvAlgo.Native
        #                             )
        
        self.avg_pool = spconv.SubMConv3d(
                                        out_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=False,
                                        indice_key=indice_key + "_avg_pool",
                                        # subm_torch=False,
                                        algo=ConvAlgo.Native
                                    )

        new_weight = torch.ones_like(self.avg_pool.weight) / (kernel_size**3)
        channel_idx = torch.arange(out_channels).cuda()
        channel_mask = torch.eye(max(channel_idx) + 1)[channel_idx].view(1, 1, 1, out_channels, out_channels) # convert 2 onhot
        # print("new_weight shape:", new_weight.shape, "channel_mask shape:", channel_mask.shape)
        # assert False
        new_weight = new_weight * channel_mask
        self.avg_pool.weight = torch.nn.parameter.Parameter(new_weight)
        self.avg_pool.weight.requires_grad = False
        
        # self._indice_list = []
        self.sigmoid = nn.Sigmoid()

    def _combine_feature(self, x_conv, x_avg, mask_position):
        # print("x_conv:", x_conv.features.requires_grad, "x_avg:",x_avg.features.requires_grad)
        # print("mask_position shape:", mask_position.shape, "ratio:", self.pruning_ratio)
        new_features = x_conv.features
        new_features[mask_position] = x_avg.features[mask_position]
        x_conv = x_conv.replace_feature(new_features)
        return x_conv
 
    def forward(self, x_conv):
        # pred importance
        x_conv_predict = x_conv
        x_conv_predict = self.pred_conv(x_conv_predict)
        voxel_importance = self.sigmoid(x_conv_predict.features) # [N, 1]
        # print("voxel_importance shape:", voxel_importance.shape, "num:", int(voxel_importance.shape[0]*self.pruning_ratio))
        # get mask
        mask_position = torch.argsort(voxel_importance.view(-1,))[:int(voxel_importance.shape[0]*self.pruning_ratio)]
        # conv
        x_conv = x_conv.replace_feature(x_conv.features * voxel_importance)
        x_avg = self.avg_pool(x_conv)
        x_conv = self.conv_block(x_conv)
        # mask feature
        x_conv = self._combine_feature(x_conv, x_avg, mask_position)
        return x_conv

class DynamicPruningConvNoPool(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, indice_key=None, bias=False, pruning_ratio=0.5):
        super(DynamicPruningConvNoPool, self).__init__()
        self.pruning_ratio = pruning_ratio
        self.kernel_size = kernel_size
        # self.kernel_size_dst = kernel_size_dst
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.pred_conv = spconv.SubMConv3d(
                in_channels,
                1,
                kernel_size=kernel_size,
                stride=1,
                padding=int(kernel_size//2),
                bias=False,
                indice_key=indice_key + "_pred_conv",
                algo=ConvAlgo.Native
            )

        self.conv_block = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=bias,
                                        indice_key=indice_key,
                                        # subm_torch=False,
                                        algo=ConvAlgo.Native
                                    )
        
        # self.channel_align = spconv.SubMConv3d(
        #                                 in_channels,
        #                                 out_channels,
        #                                 kernel_size=1,
        #                                 stride=1,
        #                                 bias=bias,
        #                                 padding=0,
        #                                 indice_key=indice_key + "_channel_align",
        #                                 # subm_torch=False,
        #                                 algo=ConvAlgo.Native
        #                             )
        self.sigmoid = nn.Sigmoid()

    def _convert_indice_dict(self, indice_dict, mask_position):
        indice_dict_pos = copy.deepcopy(indice_dict)
        indice_dict_neg = copy.deepcopy(indice_dict)
        _mask = ~torch.isin(indice_dict_pos.indice_pairs[1], mask_position)
        indice_dict_pos.indice_pairs[1][_mask] = -1
        indice_dict_neg.indice_pairs[1][~_mask] = -1
        # indice_dict_pos.indice_pair_num = 
        return False

    def _combine_feature(self, x_conv, x_notim, mask_position):
        # print("x_conv:", x_conv.features.requires_grad, "x_avg:",x_avg.features.requires_grad)
        # print("mask_position shape:", mask_position.shape, "ratio:", self.pruning_ratio)
        new_features = x_conv.features
        new_features[mask_position] = x_notim.features[mask_position]
        x_conv = x_conv.replace_feature(new_features)
        return x_conv
 
    def forward(self, x_conv):
        # pred importance
        x_conv_predict = x_conv
        x_conv_predict = self.pred_conv(x_conv_predict)
        voxel_importance = self.sigmoid(x_conv_predict.features) # [N, 1]
        # print("voxel_importance shape:", voxel_importance.shape, "num:", int(voxel_importance.shape[0]*self.pruning_ratio))
        # get mask
        mask_position = torch.argsort(voxel_importance.view(-1,))[:int(voxel_importance.shape[0]*self.pruning_ratio)]
        # conv
        # print("self.avg_pool.weight:", self.avg_pool.weight)
        x_conv = x_conv.replace_feature(x_conv.features * voxel_importance)
        x_notim = x_conv
        x_conv = self.conv_block(x_conv)
        # mask feature
        x_conv = self._combine_feature(x_conv, x_notim, mask_position)
        return x_conv

class DynamicPruningConvLinearPred(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, indice_key=None, bias=False, pruning_ratio=0.5):
        super(DynamicPruningConvLinearPred, self).__init__()
        self.pruning_ratio = pruning_ratio
        self.kernel_size = kernel_size
        # self.kernel_size_dst = kernel_size_dst
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.pred_conv = spconv.SubMConv3d(
                in_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=int(kernel_size//2),
                bias=False,
                indice_key=indice_key + "_pred_conv",
                algo=ConvAlgo.Native
            )

        self.conv_block = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=bias,
                                        indice_key=indice_key,
                                        # subm_torch=False,
                                        algo=ConvAlgo.Native
                                    )
        
        # self.channel_align = spconv.SubMConv3d(
        #                                 in_channels,
        #                                 out_channels,
        #                                 kernel_size=1,
        #                                 stride=1,
        #                                 bias=bias,
        #                                 padding=0,
        #                                 indice_key=indice_key + "_channel_align",
        #                                 # subm_torch=False,
        #                                 algo=ConvAlgo.Native
        #                             )
        
        self.avg_pool = spconv.SubMConv3d(
                                        out_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=False,
                                        indice_key=indice_key + "_avg_pool",
                                        # subm_torch=False,
                                        algo=ConvAlgo.Native
                                    )
        new_weight = torch.ones_like(self.avg_pool.weight) / (kernel_size**3)
        channel_idx = torch.arange(out_channels).cuda()
        channel_mask = torch.eye(max(channel_idx) + 1)[channel_idx].view(1, 1, 1, out_channels, out_channels) # convert 2 onhot
        # print("new_weight shape:", new_weight.shape, "channel_mask shape:", channel_mask.shape)
        # assert False
        new_weight = new_weight * channel_mask
        self.avg_pool.weight = torch.nn.parameter.Parameter(new_weight)
        self.avg_pool.weight.requires_grad = False
        
        # self._indice_list = []
        self.sigmoid = nn.Sigmoid()

    def _convert_indice_dict(self, indice_dict, mask_position):
        indice_dict_pos = copy.deepcopy(indice_dict)
        indice_dict_neg = copy.deepcopy(indice_dict)
        _mask = ~torch.isin(indice_dict_pos.indice_pairs[1], mask_position)
        indice_dict_pos.indice_pairs[1][_mask] = -1
        indice_dict_neg.indice_pairs[1][~_mask] = -1
        # indice_dict_pos.indice_pair_num = 
        return False

    def _combine_feature(self, x_conv, x_avg, mask_position):
        # print("x_conv:", x_conv.features.requires_grad, "x_avg:",x_avg.features.requires_grad)
        # print("mask_position shape:", mask_position.shape, "ratio:", self.pruning_ratio)
        new_features = x_conv.features
        new_features[mask_position] = x_avg.features[mask_position]
        x_conv = x_conv.replace_feature(new_features)
        return x_conv
 
    def forward(self, x_conv):
        # pred importance
        x_conv_predict = x_conv
        x_conv_predict = self.pred_conv(x_conv_predict)
        voxel_importance = self.sigmoid(x_conv_predict.features) # [N, 1]
        # print("voxel_importance shape:", voxel_importance.shape, "num:", int(voxel_importance.shape[0]*self.pruning_ratio))
        # get mask
        mask_position = torch.argsort(voxel_importance.view(-1,))[:int(voxel_importance.shape[0]*self.pruning_ratio)]
        # conv
        x_conv = x_conv.replace_feature(x_conv.features * voxel_importance)
        x_avg = self.avg_pool(x_conv)
        x_conv = self.conv_block(x_conv)
        # mask feature
        x_conv = self._combine_feature(x_conv, x_avg, mask_position)
        return x_conv

class DynamicPruningConvAttnPred(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, indice_key=None, bias=False, pruning_ratio=0.5):
        super(DynamicPruningConvAttnPred, self).__init__()
        self.pruning_ratio = pruning_ratio
        self.kernel_size = kernel_size
        # self.kernel_size_dst = kernel_size_dst
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # self.pred_conv = spconv.SubMConv3d(
        #         in_channels,
        #         1,
        #         kernel_size=kernel_size,
        #         stride=1,
        #         padding=int(kernel_size//2),
        #         bias=False,
        #         indice_key=indice_key + "_pred_conv",
        #         algo=ConvAlgo.Native
        #     )

        self.conv_block = spconv.SubMConv3d(
                                        in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=bias,
                                        indice_key=indice_key,
                                        # subm_torch=False,
                                        algo=ConvAlgo.Native
                                    )
        
        # self.channel_align = spconv.SubMConv3d(
        #                                 in_channels,
        #                                 out_channels,
        #                                 kernel_size=1,
        #                                 stride=1,
        #                                 bias=bias,
        #                                 padding=0,
        #                                 indice_key=indice_key + "_channel_align",
        #                                 # subm_torch=False,
        #                                 algo=ConvAlgo.Native
        #                             )
        
        self.avg_pool = spconv.SubMConv3d(
                                        out_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=int(kernel_size//2),
                                        bias=False,
                                        indice_key=indice_key + "_avg_pool",
                                        # subm_torch=False,
                                        algo=ConvAlgo.Native
                                    )
        new_weight = torch.ones_like(self.avg_pool.weight) / (kernel_size**3)
        channel_idx = torch.arange(out_channels).cuda()
        channel_mask = torch.eye(max(channel_idx) + 1)[channel_idx].view(1, 1, 1, out_channels, out_channels) # convert 2 onhot
        new_weight = new_weight * channel_mask
        self.avg_pool.weight = torch.nn.parameter.Parameter(new_weight)
        self.avg_pool.weight.requires_grad = False
        
        self.sigmoid = nn.Sigmoid()

    def _combine_feature(self, x_conv, x_avg, mask_position):
        new_features = x_conv.features
        new_features[mask_position] = x_avg.features[mask_position]
        x_conv = x_conv.replace_feature(new_features)
        return x_conv
 
    def forward(self, x_conv):
        # pred importance with attn_map
        x_conv_feature = x_conv.features
        x_conv_attn_map = torch.abs(x_conv_feature).sum(1) / x_conv_feature.shape[1]
        voxel_importance = self.sigmoid(x_conv_attn_map.view(-1, 1))
        # print("voxel_importance:", voxel_importance.shape, "voxel_importance max:", voxel_importance.max(), "voxel_importance min:", voxel_importance.min())
        mask_position = torch.argsort(voxel_importance.view(-1,))[:int(voxel_importance.shape[0]*self.pruning_ratio)]
        # conv
        x_conv = x_conv.replace_feature(x_conv.features * voxel_importance)
        x_avg = self.avg_pool(x_conv)
        x_conv = self.conv_block(x_conv)
        # mask feature
        x_conv = self._combine_feature(x_conv, x_avg, mask_position)
        return x_conv
