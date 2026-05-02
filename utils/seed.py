import random
import numpy as np
import torch
import os

def set_global_seed(seed: int = 42):
    """
    Khóa toàn bộ các bộ sinh số ngẫu nhiên để đảm bảo tính tái lập (Reproducibility)
    cho hệ thống AI và cấu trúc mạng giao thông.
    """
    # 1. Khóa seed của Python cơ bản và hash môi trường
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Khóa seed của Numpy (dùng cho mảng trạng thái và epsilon-greedy)
    np.random.seed(seed)
    
    # 3. Khóa seed của PyTorch (dùng cho khởi tạo trọng số mạng DQN/PPO)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Ép cuDNN chạy theo chế độ deterministic (tuy có thể làm giảm nhẹ 
        # tốc độ train nhưng đảm bảo kết quả chính xác 100% giữa các lần chạy)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"[INFO] Đã khóa toàn bộ Global Seed ở mốc: {seed}")