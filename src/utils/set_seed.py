import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Cố định seed cho toàn bộ pipeline để đảm bảo kết quả có thể tái tạo.
    """
    # 1. Python built-in random module
    random.seed(seed)

    # 2. Python hash seed (cố định thứ tự sinh hash của dictionary, set)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3. Numpy
    np.random.seed(seed)

    # 4. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Nếu dùng multi-GPU

    # 5. CUDNN (Cực kỳ quan trọng cho Convolutional Neural Networks)
    # deterministic = True: Bắt buộc cuDNN dùng các thuật toán mang tính tái tạo (có thể làm chậm tốc độ train đi một chút)
    torch.backends.cudnn.deterministic = True
    # benchmark = False: Ngăn cuDNN tự động tìm kiếm thuật toán convolution tối ưu nhất cho phần cứng (vì nó sinh ra tính ngẫu nhiên)
    torch.backends.cudnn.benchmark = False

    print(f"[*] Đã set global seed = {seed} cho Python, Numpy, PyTorch và cuDNN.")
