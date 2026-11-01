"""Random-state handling shared by the stochastic model stages."""

import os
import random

import numpy as np


def seed_everything(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
