# Copyright 2024 Hirokazu Murayama
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This code is based on software developed by Proxima Technology Inc, TIER IV
# and licensed under the Apache License, Version 2.0.

import numpy as np
import matplotlib.pyplot as plt
import time
import drive_setting
from car_functions import F, F_diff
from my_drive_control import iLQR_drive, MPPI_drive

actual_state_dim = drive_setting.actual_state_dim
actual_input_dim = drive_setting.actual_input_dim
nx = drive_setting.nx
nu = drive_setting.nu
N = drive_setting.N
dt = drive_setting.dt

ctrl_0 = iLQR_drive()
ctrl = MPPI_drive()
x = np.zeros((nx,))

start = time.time()
u = ctrl_0.control(x_current=x)
x = F(x,u,dt)
print(x)
print(u)
ctrl.u = ctrl_0.u
for k in range(99):
    u = ctrl.control(x_current=x)
    x = F(x,u,dt)
    print(x)
    print(u)
end = time.time()

print(end-start)