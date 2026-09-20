import random
import numpy as np
import os
import torch

def seed_cpu(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    torch.manual_seed(seed)
    
    # Зажимаем вычисления в 1 поток (убирает разброс округления Float32 в OpenMP/MKL)
    torch.set_num_threads(1)
    
    # Принудительно включаем детерминированные алгоритмы
    torch.use_deterministic_algorithms(True, warn_only=True)