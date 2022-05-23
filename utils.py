import os
import sys
import random
import numpy as np
import pandas as pd
from typing import *

import warnings
warnings.filterwarnings('ignore')

class ddict(dict):
    """using dot instead of brackets to access dictionary item"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def seed_everything(seed=20, use_torch=True):
    """set seed for all

    Args:
        seed (int, optional): seed value. Defaults to 20.
        use_torch (bool, optional): whether pyTorch is in use. Defaults to True.
    """ 
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True