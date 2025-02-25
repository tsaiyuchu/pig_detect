import torch
import torch.nn as nn

# 測試 3D CNN 是否可用
model = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=1, padding=1).cuda()
input_data = torch.randn(1, 3, 16, 112, 112).cuda()  # batch_size, channels, depth, height, width
output = model(input_data)
print(output.shape)

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
