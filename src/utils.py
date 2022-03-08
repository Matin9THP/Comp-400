"""
General utility functions. Taken from: https://github.com/cs230-stanford/
cs230-code-examples/tree/master/tensorflow/vision
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging


class Params():
    """
    Class that loads hyperparameters from a json file.
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """
        Saves parameters to json file
        """
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """
        Loads parameters from json file
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """
        Gives dict-like access to Params instance by
        `params.dict['learning_rate']`
        """
        return self.__dict__


def set_logger():
    """
    set the logger to log info in terminal
    """
    logger = logging.getLogger('pytorchUnNet3D')
    logger.setLevel(logging.DEBUG)
    
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


    return logger
