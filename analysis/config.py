import json
from copy import deepcopy
from glob import glob
import numpy as np
import os
from utils import utils


class Config:

    def __init__(self, cfg_dict):
        self.data = cfg_dict['data']
        self.train = cfg_dict['train']
        self.model = cfg_dict['model']

    def device(self):
        return 'cpu'

    @property
    def dataset_name(self):
        return "-".join(self.data['matf'][:-4].split('_')[2:])


