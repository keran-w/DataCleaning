import os, sys, re
import random
import numpy as np
import pandas as pd
from io import StringIO
from typing import *

from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

class ddict(dict):
    """using dot instead of brackets to access dictionary item"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def seed_everything(seed=20):
    import torch
    """set seed for all"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True