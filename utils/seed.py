import random
import numpy as np
import torch
import os

def set_global_seed(seed: int = 42, deterministic: bool = True):
    """
    Khóa toàn bộ các bộ sinh số ngẫu nhiên để đảm bảo tính tái lập (Reproducibility)
    cho hệ thống AI và cấu trúc mạng giao thông.
    """
    # 1. Khóa seed của Python cơ bản và hash môi trường
    # PYTHONHASHSEED is most reliable when also set before process startup,
    # but setting it here keeps subprocesses and logged config consistent.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    
    # 2. Khóa seed của Numpy (dùng cho mảng trạng thái và epsilon-greedy)
    np.random.seed(seed)
    
    # 3. Khóa seed của PyTorch (dùng cho khởi tạo trọng số mạng DQN/PPO)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Ép PyTorch/cuDNN ưu tiên thuật toán deterministic.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        
    print(f"[INFO] Đã khóa toàn bộ Global Seed ở mốc: {seed} | deterministic={deterministic}")
