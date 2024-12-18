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
from . import drive_setting
from typing import Tuple, Callable
from .numba_iLQR_functions import backward_pass, forward_pass, rollout, cost_calc
from .numba_MPPI_functions import l_cost_without_input_quad, compute_optimal_input


"""共通のパラメータ"""
actual_state_dim = drive_setting.actual_state_dim
actual_input_dim = drive_setting.actual_input_dim
nx = drive_setting.nx
nu = drive_setting.nu
N = drive_setting.N
dt = drive_setting.dt

"""MPPI用のパラメータ"""
lam = drive_setting.lam
sample_num = drive_setting.sample_num


class Controller:
    """mother class of controller (for model predictive control)"""
    
    def __init__(self):
        """get parameter"""
        pass

    def control(self):
        """do control calculation"""
        pass



class iLQR_drive(Controller):
    """class for iLQR"""

    def __init__(
            self,
            ilqr_step: int = 10, # Backward Pass と Forward Pass を何回繰り返すか
    ):
        super().__init__()
        self.ilqr_step = ilqr_step
        self.u = np.zeros((N,nu))

    def pred_weights(
            self,
            x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Qは固定なので、R,R_acc,R_jerkを計算する"""
        """後々ここはガウス過程を用いる様に変更する"""
        q = 1*np.diag(np.array([1, 1, 0], dtype=np.float64))
        Q = np.stack([q]*N, dtype=np.float64)
        R = 1*np.stack([np.eye(nu,dtype=np.float64)]*N)
        R_acc = 1e-2*np.stack([np.eye(nu,dtype=np.float64)]*N)
        R_jerk = 1e-4*np.stack([np.eye(nu,dtype=np.float64)]*N)

        return Q, R, R_acc, R_jerk
    
    def get_path_or_destination(
            self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """パスフォロワーにするときはこの関数を改造する"""
        x_des = np.stack([np.array([5,4,0])]*(N+1))
        u_des = np.stack([np.zeros(nu)]*N)

        return x_des, u_des

    def control(
            self,
            x_current: np.ndarray, # ここでは現在座標
    ):
        u = self.u
        ilqr_step = self.ilqr_step

        """初期解で状態変数を予測"""
        x = rollout(x_current,u,dt)

        """パスを入手する"""
        x_des, u_des = self.get_path_or_destination()

        """今回はテストなので、最初に重みを計算してしまう"""
        Q,R,R_acc,R_jerk = self.pred_weights(x)

        """現状のコストを計算"""
        cost_original = cost_calc(x,u,x_des,u_des,Q,R,R_acc,R_jerk,dt)

        for step in range(ilqr_step):
            K, d, del_V1, del_V2 = backward_pass(x,u,x_des,u_des,Q,R,R_acc,R_jerk,dt)
            x, u, cost = forward_pass(x,u,x_des,u_des,Q,R,R_acc,R_jerk,K,d,del_V1,del_V2,dt)
            
            if np.abs(cost-cost_original) < 1.0:
                break

            cost_original = cost

        self.u = u
        
        return u[0]
    


class MPPI_drive(Controller):
    """class for MPPI"""

    def __init__(
            self,
            mppi_step: int = 1,  # MPPIを何回繰り返すか
            lam: float = lam,
            sample_num: int = sample_num,
    ):
        super().__init__()
        self.u = np.zeros((N,nu))
        self.mppi_step = mppi_step
        self.lam = lam
        self.sample_num = sample_num

        """拘束条件はここで定義しておく"""
        self.u_max , self.u_min = self.get_input_constraint()

    def pred_weights(
            self,
            x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Qは固定なので、R,R_acc,R_jerkを計算する"""
        """後々ここはガウス過程を用いる様に変更する"""
        q = 1*np.diag(np.array([1, 1, 0], dtype=np.float64))
        Q = np.stack([q]*N, dtype=np.float64)
        R = 1*np.stack([np.eye(nu,dtype=np.float64)]*N)
        R_acc = 1e-2*np.stack([np.eye(nu,dtype=np.float64)]*N)
        R_jerk = 1e-4*np.stack([np.eye(nu,dtype=np.float64)]*N)

        return Q, R, R_acc, R_jerk
    
    def get_path_or_destination(
            self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """パスフォロワーにするときはこの関数を改造する"""
        x_des = np.stack([np.array([5,4,0])]*(N+1))
        u_des = np.stack([np.zeros(nu)]*N)

        return x_des, u_des
    
    def get_input_constraint(
            self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """入力の拘束条件を得る関数"""
        u_max = np.array([0.7, np.pi])
        u_min = np.array([-0.7, -np.pi])
        return u_max, u_min
    
    def compute_sigma(
            self,
            R: np.ndarray,
    ) -> np.ndarray:
        """入力の重み行列からシグマ（の逆行列）を計算する関数"""
        """今はこれだけ"""
        sigma = R/lam
        return sigma
    
    def control(
            self,
            x_current: np.ndarray,
    ) -> np.ndarray:
        original_input = self.u
        mppi_step = self.mppi_step
        lam = self.lam
        sample_num = self.sample_num

        """初期解で状態変数を予測"""
        Traj = rollout(x_current,original_input,dt)

        """パスを入手する"""
        x_des, u_des = self.get_path_or_destination()

        """今回はテストなので、最初に重みを計算してしまう"""
        Q,R,R_acc,R_jerk = self.pred_weights(Traj)

        """シグマの計算"""
        sigma = self.compute_sigma(R)

        """今は拘束条件は外付けで"""
        u_max = self.u_max
        u_min = self.u_min

        for step in range(mppi_step):
            optimal_input, optimal_cost = compute_optimal_input(x_current,original_input,x_des,u_des,u_max,u_min,Q,R_acc,R_jerk,sigma,sample_num,lam,0,dt)
            """今は途中でのbreakとか考えずに実装する"""
            original_input = optimal_input
            original_cost = optimal_cost
        
        self.u = optimal_input

        return optimal_input[0]

    

if __name__ == "__main__":
    pass