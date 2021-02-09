import math
import numpy as np
import torch

NEG_INF = torch.finfo(torch.float32).min
MATH_PI = math.pi
EPSILON = np.finfo(np.float32).eps
MAX_INT16 = np.iinfo(np.int16).max
SOUND_SPEED = 343  # m/s
