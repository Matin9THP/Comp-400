"""
Data feeding function for train and test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from data_utils import Dataset


def input_fn(params):
    """
    Simple input_fn for our 3D U-Net estimator, handling train and test data
    preparation.

    Args:
        params (dict): Params for setting up the data. Expected keys are:
            max_scans (int): Maximum number of scans we see in any patient.
            train_img_size (int): Width and height of resized training images.
            batch_size (int): Number of of patient in each batch for training.
            num_classes (int): Number of mutually exclusive output classes.
            train_dataset_path (str): Path to pickled
                :class:`src.data_utils.Dataset` object.
            test_dataset_path (str): Path to pickled
                :class:`src.data_utils.Dataset` object.

    Returns:
    """
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # for training we use a batch number and pad each 3D scan to have equal
    # depth, width and height have already been set to 128 in preprocessing
    max_s = params['max_scans']
    w = h = params['train_img_size']

    train_loader = Dataset.load_dataset(
        os.path.join(package_root, params['train_dataset_path'])
    ).create_pytorch_dataset()

    print("################")

    test_loader = Dataset.load_dataset(
        os.path.join(package_root, params['test_dataset_path'])
    ).create_pytorch_dataset()

    return train_loader, test_loader
