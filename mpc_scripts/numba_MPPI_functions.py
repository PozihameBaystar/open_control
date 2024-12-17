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
from typing import Tuple, Callable
from numba import njit, prange

import drive_setting
from car_functions import F, F_diff

"""
MPPIの関数を実装する
とりあえず自分の仕様で書く
"""

actual_state_dim = drive_setting.actual_state_dim
actual_input_dim = drive_setting.actual_input_dim
nx = actual_state_dim + 2*actual_input_dim
nu = actual_input_dim
N = drive_setting.N
dt = drive_setting.dt
Th = N*dt

"""
MPPI用の変数
"""
lam = drive_setting.lam
sample_num = drive_setting.sample_num


"""入力の二次形式の部分を除いた部分のステージコスト"""
@njit(cache=True,fastmath=True)
def l_cost_without_input_quad(
    x: np.ndarray,  # こちらは過去の入力等も含む拡張状態変数
    u: np.ndarray,
    x_des: np.ndarray,
    u_des: np.ndarray,
    Q: np.ndarray,
    R_acc: np.ndarray,
    R_jerk: np.ndarray,
    dt: float = dt,
) -> float:
    acc = (u-x[actual_state_dim:actual_state_dim+actual_input_dim])/dt
    jerk = (u- 2*x[actual_state_dim:actual_state_dim+actual_input_dim] + x[actual_state_dim+actual_input_dim:actual_state_dim+2*actual_input_dim])/dt**2

    cost = 0.5*(x[:actual_state_dim]-x_des).T @ Q @ (x[:actual_state_dim]-x_des) \
        + 0.5*acc.T @ R_acc @ acc + 0.5*jerk.T @ R_jerk @ jerk
    cost = cost*dt
    return cost


@njit(cache=True,fastmath=True,parallel=True)
def generate_sampled_inputs(
    original_inputs: np.ndarray,  # これは前回の制御入力
    sigma: np.ndarray,  # N × actual_input_dim × actual_input_dim
    u_max: np.ndarray,  # この次元は単にnu
    u_min: np.ndarray,
    alpha: float,
    sample_num: int = sample_num,
) -> np.ndarray:
    """制御入力をサンプリングする関数"""
    N = original_inputs.shape[0]
    sampled_inputs = np.zeros((sample_num,N,nu))  # サンプルの数だけ作る

    for k in prange(sample_num):
        sampled_inputs[k] = original_inputs
        if k < (1-alpha)*sample_num:
            for i in range(N):
                cov = np.linalg.inv(sigma[i])
                rams = np.zeros(nu)
                for j in range(nu):
                    ram = np.random.normal(loc=0,scale=np.sqrt(cov[j,j]),size=1)[0]
                    rams[j] = ram
                sampled_inputs[k,i] += rams
        else:
            for i in range(N):
                cov = np.linalg.inv(sigma[i])
                rams = np.zeros(nu)
                for j in range(nu):
                    ram = np.random.normal(loc=0,scale=np.sqrt(cov[j,j]),size=1)[0]
                    rams[j] = ram
                sampled_inputs[k,i] = rams

    #for i in range(nu):
    #    sampled_inputs[:,:,i] = np.clip(sampled_inputs[:,:,i],u_min[i],u_max[i])

    return sampled_inputs  # 次元数はサンプルの数sample_num×ホライズンステップN×入力次元nu


@njit(cache=True,fastmath=True)
def calc_forward_with_cost(
    x_current: np.ndarray,
    inputs: np.ndarray,
    original_inputs: np.ndarray,
    x_des: np.ndarray,  # ステージごとに異なる目標値を設定できるようにしておく（iLQRの方と同じ）
    u_des: np.ndarray,
    Q: np.ndarray,
    R_acc: np.ndarray,
    R_jerk: np.ndarray,
    sigma: np.ndarray,  # 実際にはシグマの逆行列
    lam: float = lam,
    dt: float = dt,
) -> Tuple[np.ndarray, float]:
    """現在の状態変数と制御入力から将来の状態変数とコスト（入力の二次形式を除いた部分）を計算"""
    N = inputs.shape[0]
    Traj = np.zeros((N+1,nx))
    Traj[0] = x_current
    cost = 0

    for i in range(N):
        cost += l_cost_without_input_quad(
            x=Traj[i],
            u=inputs[i],
            x_des=x_des[i],
            u_des=u_des[i],
            Q=Q[i],
            R_acc=R_acc[i],
            R_jerk=R_jerk[i],
            dt=dt,
        ) \
        + lam * (original_inputs[i]-u_des[i]).T @ sigma[i] @ inputs[i]
        Traj[i+1] = F(
            x=Traj[i],
            u=inputs[i],
            dt=dt,
        )
    
    return Traj, cost


@njit(cache=True,fastmath=True,parallel=True)
def calc_forward_with_cost_for_candidates(
    x_current: np.ndarray,
    original_inputs: np.ndarray,
    sampled_inputs: np.ndarray,
    x_des: np.ndarray,
    u_des: np.ndarray,
    Q: np.ndarray,
    R_acc: np.ndarray,
    R_jerk: np.ndarray,
    sigma: np.ndarray,
    sample_num: int = sample_num,
    lam: float = lam,
    dt: float = dt,
) -> Tuple[np.ndarray, np.ndarray]:
    Trajs = np.zeros((sample_num,N+1,nx))  # 軌道を格納する変数
    costs = np.zeros(sample_num)  # コストを格納する変数
    for k in prange(sample_num):
        Trajs[k], costs[k] = calc_forward_with_cost(x_current,sampled_inputs[k],original_inputs,x_des,u_des,Q,R_acc,R_jerk,sigma,lam,dt)
    return Trajs, costs


def compute_optimal_input(
    x_current: np.ndarray,
    original_inputs: np.ndarray,
    x_des: np.ndarray,
    u_des: np.ndarray,
    u_max: np.ndarray,
    u_min: np.ndarray,
    Q: np.ndarray,
    R_acc: np.ndarray,
    R_jerk: np.ndarray,
    sigma: np.ndarray,
    sample_num: int = sample_num,
    lam: float = lam,
    alpha: float = 0,
    dt: float = dt,
) -> Tuple[np.ndarray, float]:
    N = original_inputs.shape[0]

    """まずはサンプルを生成する"""
    sampled_inputs = generate_sampled_inputs(original_inputs,sigma,u_max,u_min,alpha,sample_num)
    
    """次に、各サンプルごとにforward計算とコスト計算を行う"""
    Trajs = np.zeros((sample_num,N+1,nx))  # 軌道を格納する変数
    costs = np.zeros(sample_num)  # コストを格納する変数
    Trajs, costs = calc_forward_with_cost_for_candidates(x_current,original_inputs,sampled_inputs,x_des,u_des,Q,R_acc,R_jerk,sigma,sample_num,lam,dt)
    
    """コストから重みを計算する"""
    best_cost = costs.min()
    weights = np.exp(-(costs-best_cost)/lam)
    weights = weights/weights.sum()
    """最適な入力値の計算"""
    optimal_input = (sampled_inputs.T @ weights).T
    """コストの計算"""
    optimal_cost = np.dot(weights,costs)

    return optimal_input, optimal_cost


if __name__ == "__main__":
    pass