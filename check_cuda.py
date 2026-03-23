import torch

print('PyTorch 版本:', torch.__version__)
print('CUDA 可用:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA 版本:', torch.version.cuda)
else:
    print('警告：CUDA 不可用，訓練會非常慢！')
