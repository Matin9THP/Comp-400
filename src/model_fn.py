"""
3D U-Net network, for prostate MRI scans.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
from network import UNet3D


def loss_fn(loss ,logits, labels):
    """
    Args:
        logits (torch.Tensor): Predicted logits.
        labels (torch.Tensor): Ground truth labels.
        loss (torch.nn.CrossEntropyLoss): Loss function.
    Returns:
        loss (torch.Tensor): Loss tensor.
    """
    return loss(logits.float(), labels.float())



def model_fn(params, logger, in_cuda):
    """
    get model and optimizer
    Args:
        params (Params): Hyperparameters for training.

    Returns:
        model (torch.nn.Module): The model.
    """
    # -------------------------------------------------------------------------
    # get logits from 3D U-Net
    # -------------------------------------------------------------------------
    unetLogits = UNet3D(params, logger)
    if in_cuda:
        unetLogits.cuda()


    # -------------------------------------------------------------------------
    # optimizer for training
    # -------------------------------------------------------------------------
    optimizer = optim.Adam(unetLogits.parameters(), lr=params['learning_rate'],betas=(0.5, 0.999))

    return unetLogits, optimizer
