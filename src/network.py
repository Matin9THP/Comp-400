"""
Simple implementation of 3D U-Net building function.
Original paper: https://arxiv.org/abs/1606.06650
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn

log = logging.getLogger('pytorch')\

def create_name(part, layer, i):
    """
    Helper function for generating names for layers.

    Args:
        part (str): Part/path the layer belongs to.
        layer (str): The function of the layer, .e.g conv3d.
        i (int): The layer depth of the layer.

    Returns:
        str: Concatenated layer name.
    """
    return "%s_%s_l%d" % (part, layer, i)

def conv3d_bn_relu(in_dim, filters, kernel, stride, padding,
                   batch_norm, part, layer_depth):
    """
    Basic conv3d > Batch Normalisation > Relu building block for the network.

    Args:
        in_dim (int): See conv3D torch.
        filters (int): See conv3D docs.
        kernel (int): See conv3D docs.
        strides (int): See conv3D docs.
        padding (str): See conv3D docs.
        batch_norm (bool): Whether to use batch_norm in the conv3d blocks.
        part (str): Needed for name generation.
        layer_depth (int): Needed for name generation.


    Returns:
    """

    if batch_norm:
        conv3d_bn_relu = nn.Sequential(
            nn.Conv3d(
                in_dim, #TODO set input dimension
                filters, 
                kernel, 
                stride=stride, 
                padding=padding, 
                bias=False),
            nn.BatchNorm3d(filters, affine= True),
            nn.LeakyReLU(0.2, inplace=True))
    else:
        conv3d_bn_relu = nn.Sequential(
            nn.Conv3d(
                in_dim, #TODO set input dimesnion
                filters, 
                kernel, 
                stride=stride, 
                padding=padding, 
                bias=False),
            nn.LeakyReLU(0.2, inplace=True))
    return conv3d_bn_relu





class UNet3D(nn.Module):
    def __init__(self, params, log):
        super().__init__()

        self.log = log

        # -------------------------------------------------------------------------
        # setup
        # -------------------------------------------------------------------------

        # extract model params
        self.depth = params['depth']
        self.n_base_filters = params['n_base_filters']
        self.num_classes = params['num_classes']
        self.batch_norm = params['batch_norm']

        # additional model params that are currently baked into the model_fn
        conv_size = 3
        conv_strides = 1
        pooling_size = 2
        pooling_strides = 2
        padding = 'same'
        padding = 1




        # variables to track architecture building with loops
        concat_layer_sizes = list()
        next_layer_size = 1


        # adding layers in levels
        self.analysis_levels = {}
        self.synthesis_levels = {}

        # -------------------------------------------------------------------------
        # network architecture: analysis path
        # -------------------------------------------------------------------------

        for layer_depth in range(self.depth):
            # two conv3d > batch norm > relu blocks
            conv1_filters = self.n_base_filters * (2 ** layer_depth)
            conv2_filters = conv1_filters * 2


            self.log.info('next_layer: %s' % next_layer_size)



            layer1 = conv3d_bn_relu(
                in_dim=next_layer_size,
                filters=conv1_filters,
                kernel=conv_size,
                stride=conv_strides,
                padding=padding,
                batch_norm=self.batch_norm,
                part='analysis',
                layer_depth=layer_depth,
            )
            self.log.info('conv1: %s' % conv1_filters)

            layer2 = conv3d_bn_relu(
                in_dim=conv1_filters,
                filters=conv2_filters,
                kernel=conv_size,
                stride=conv_strides,
                padding=padding,
                batch_norm=self.batch_norm,
                part='analysis',
                layer_depth=layer_depth,
            )
            concat_layer_sizes.append(conv2_filters)
            self.log.info('conv1: %s' % conv2_filters)

            # add max pooling unless we're at the end of the bottleneck
            if layer_depth < self.depth - 1:
                max_layer = nn.MaxPool3d(
                    pooling_size,
                    stride=pooling_strides,
                    padding=0
                )
                
                # log.info('maxpool layer: %s' % max_layer)
                self.analysis_levels[layer_depth] = [layer1, layer2, max_layer]
            else:
                self.analysis_levels[layer_depth] = [layer1, layer2]

            next_layer_size = conv2_filters

        # -------------------------------------------------------------------------
        # network architecture: synthesis path
        # -------------------------------------------------------------------------

        for layer_depth in range(self.depth - 2, -1, -1):
            # add up-conv
            n_filters = self.n_base_filters * (2 ** layer_depth) * 2
            up_conv = nn.ConvTranspose3d(
                next_layer_size, 
                n_filters * 2,
                pooling_size,
                stride=pooling_strides,
                padding=0,
                bias=False
            )
            # log.info('upconv layer: %s %s' % (n_filters, layer_depth))
            self.log.info('concat_layer input layer: %s' % (n_filters * 3))

            # two conv3d > batch norm > relu blocks
            layer1 = conv3d_bn_relu(
                in_dim=n_filters * 3, 
                filters=concat_layer_sizes[layer_depth],
                kernel=conv_size,
                stride=conv_strides,
                padding=padding,
                batch_norm=self.batch_norm,
                part='synthesis',
                layer_depth=layer_depth
            )

            log.info('concat_layer output layer: %s' % (concat_layer_sizes[layer_depth]))

            layer2 = conv3d_bn_relu(
                in_dim = concat_layer_sizes[layer_depth], 
                filters=concat_layer_sizes[layer_depth],
                kernel=conv_size,
                stride=conv_strides,
                padding=padding,
                batch_norm=self.batch_norm,
                part='synthesis2',
                layer_depth=layer_depth
            )
            self.synthesis_levels[layer_depth] = [up_conv, layer1, layer2]

            next_layer_size = concat_layer_sizes[layer_depth]
            log.info('next_layer_size layer2 : %s' % next_layer_size)

        # final 1 x 1 x 1 conv3d layer
        self.logits = nn.Conv3d(
            next_layer_size,
            self.num_classes,
            kernel_size=1,
            stride=1,
            bias=True
        )
        log.info('output layer:: %s' % self.num_classes)

    def forward(self, x):
        
        result = {}
        for layer_depth in range(self.depth):
            x = self.analysis_levels[layer_depth][0](x)
            x = self.analysis_levels[layer_depth][1](x)
            result[layer_depth] = x
            
            if layer_depth < self.depth - 1:
                x = self.analysis_levels[layer_depth][2](x) 

        for layer_depth in range(self.depth - 2, -1, -1): 
            x = self.synthesis_levels[layer_depth][0](x)
            x = torch.cat([x, result[layer_depth]], dim = 1)
            x = self.synthesis_levels[layer_depth][1](x)
            x = self.synthesis_levels[layer_depth][2](x)
            

        x = self.logits(x) 
        return x

    def cuda(self):
        for layers in self.analysis_levels.values():
            for layer in layers:
                layer.cuda()
        for layers in self.synthesis_levels.values():
            for layer in layers:
                layer.cuda()

        self.logits.cuda()