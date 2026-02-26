import math
import numpy as np
import tf.transformations as tft
#.570796327 -0.87 0
#1.570796327 -2.87 0
# 1.570796327 -2.0 0
# -1.570796327 -0.87 0
# 1.570796327 -2.87 0
# -1.570796327 2.0 0
roll  = -1.570796327
pitch = 2.0
yaw   = 0.0
# URDF默认是绕 X->Y->Z
quat = tft.quaternion_from_euler(roll, pitch, yaw, 'sxyz')
print(quat)  # [x, y, z, w] 格式
# python /home/bigeast/humanoid-gym/resources/robots/bruce/rpy2quat.py