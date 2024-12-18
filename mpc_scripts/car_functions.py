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

import numpy as np
from typing import Tuple
from numba import njit

from . import drive_setting


"""
自動車の状態方程式関連はこっちに移す
"""

actual_state_dim = drive_setting.actual_state_dim
actual_input_dim = drive_setting.actual_input_dim
nx = actual_state_dim + 2*actual_input_dim
nu = actual_input_dim
N = drive_setting.N
dt = drive_setting.dt
Th = N*dt

"""車の状態方程式"""
@njit(cache=True,fastmath=True)
def F(
    x: np.ndarray,
    u: np.ndarray,
    dt: float = dt,
) -> np.ndarray:
    x_ = np.zeros(nx, dtype=np.float64)
    x_[actual_state_dim : actual_state_dim + actual_input_dim] += u
    x_[actual_state_dim + actual_input_dim : actual_state_dim + 2*actual_input_dim]\
        += x[actual_state_dim : actual_state_dim + actual_input_dim]
    theta = x[2]
    B = np.zeros((actual_state_dim,actual_input_dim), dtype=np.float64)
    B[0, 0] = np.cos(theta)
    B[1, 0] = np.sin(theta)
    B[2, 1] = 1.0
    x_[:actual_state_dim] += x[:actual_state_dim] + B @ u * dt
    return x_


"""状態方程式の微分"""
@njit(cache=True,fastmath=True)
def F_diff(
    x: np.ndarray,
    u: np.ndarray,
    dt: float = dt,
) -> Tuple[np.ndarray, np.ndarray]:
    theta = x[2]
    dFdx = np.zeros((nx,nx), dtype=np.float64)

    dFdx_ = np.zeros((actual_state_dim, actual_state_dim), dtype=np.float64)
    dFdx_[0,2] = -np.sin(theta)*u[0]*dt
    dFdx_[1,2] = np.cos(theta)*u[0]*dt
    dFdx[:actual_state_dim,:actual_state_dim] += dFdx_ + np.eye(actual_state_dim, dtype=np.float32)
    dFdx[actual_state_dim + actual_input_dim : actual_state_dim + 2*actual_input_dim , actual_state_dim : actual_state_dim + actual_input_dim] \
        += np.eye(actual_input_dim)
    
    dFdu = np.zeros((nx,nu), dtype=np.float64)
    dFdu_ = np.zeros((actual_state_dim,actual_input_dim), dtype=np.float64)
    dFdu_[0,0] = np.cos(theta)*dt
    dFdu_[1,0] = np.sin(theta)*dt
    dFdu_[2,1] = dt
    dFdu[:actual_state_dim, :actual_input_dim] += dFdu_
    dFdu[actual_state_dim : actual_state_dim + actual_input_dim, :actual_input_dim] += np.eye(actual_input_dim)
    return dFdx, dFdu

if __name__ == "__main__":
    pass