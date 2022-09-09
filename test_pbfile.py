
# 代码摘自:
# https://developers.google.com/protocol-buffers/docs/pythontutorial

import 写你编译后的py文件 as pb
import numpy as np
import cv2

mask_scale = 8

message = pb.message()

print(message)

with open("x.pb", "rb") as f:
    message.ParseFromString(f.read())
    
print(message)