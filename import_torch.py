import torch

if torch.cuda.is_available():
    print("太棒了！PyTorch已成功配置GPU。")
    print(f"当前使用的GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
else:
    print("哦，出错了。PyTorch未能找到可用的GPU。")