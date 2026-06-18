import os
import random

import ants.config
import numpy as np

DEBUG = False
"""If True, prints various diagnostic details to console, and creates intermediate images between steps."""

seed = 123
ants.config.set_ants_deterministic(True, seed)
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
np.random.seed(seed)
random.seed(seed)
