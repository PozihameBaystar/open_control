import numpy as np
import matplotlib.pyplot as plt
import time
import drive_setting
from car_functions import F, F_diff
from my_drive_control import iLQR_drive

actual_state_dim = drive_setting.actual_state_dim
actual_input_dim = drive_setting.actual_input_dim
nx = drive_setting.nx
nu = drive_setting.nu
N = drive_setting.N
dt = drive_setting.dt

ctrl = iLQR_drive()
x = np.zeros((nx,))

start = time.time()
for k in range(100):
    u = ctrl.control(x_current=x)
    x = F(x,u,dt)
    print(x)
    print(u)
end = time.time()

print(end-start)