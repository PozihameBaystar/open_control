import numpy as np
from typing import Tuple, Callable
from numba import njit

import drive_setting
from car_functions import F, F_diff

"""
まず分かりやすさの為、拡張した状態変数をデフォルトとして扱ってしまう
細かいところは置いといて、まず作る
"""

"""
過去の速度は二つ前まで保持する（ジャークを測る為）
状態変数は x_t, u_t-1, u_t-2
"""

"""
色々な変数をここに書く（後でyamlファイルか何かに移す!!）
"""
actual_state_dim = drive_setting.actual_state_dim
actual_input_dim = drive_setting.actual_input_dim
nx = actual_state_dim + 2*actual_input_dim
nu = actual_input_dim
N = drive_setting.N
dt = drive_setting.dt
Th = N*dt


"""ステージコストの計算"""
@njit(cache=True,fastmath=True)
def l_cost(
    x: np.ndarray,
    u: np.ndarray,
    x_des: np.ndarray,
    u_des: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    R_acc: np.ndarray,
    R_jerk: np.ndarray,
    dt: float = dt,
) -> float:
    acc = (u-x[actual_state_dim:actual_state_dim+actual_input_dim])/dt
    jerk = (u- 2*x[actual_state_dim:actual_state_dim+actual_input_dim] + x[actual_state_dim+actual_input_dim:actual_state_dim+2*actual_input_dim])/dt**2

    cost = 0.5*(x[:actual_state_dim]-x_des).T @ Q @ (x[:actual_state_dim]-x_des) + 0.5*(u-u_des).T @ R @ (u-u_des) \
        + 0.5*acc.T @ R_acc @ acc + 0.5*jerk.T @ R_jerk @ jerk
    cost = cost*dt
    
    return cost


"""ステージコストの微分の計算"""
@njit(cache=True,fastmath=True)
def l_grad(
    x: np.ndarray,
    u: np.ndarray,
    x_des: np.ndarray,
    u_des: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    R_acc: np.ndarray,
    R_jerk: np.ndarray,
    dt: float = dt,
) -> Tuple[np.ndarray, np.ndarray]:
    
    acc = (u - x[actual_state_dim:actual_state_dim+actual_input_dim])/dt
    jerk = (u- 2*x[actual_state_dim:actual_state_dim+actual_input_dim] + x[actual_state_dim+actual_input_dim:actual_state_dim+2*actual_input_dim])/dt**2
    

    lx = np.zeros(nx, dtype=np.float64)
    lu = np.zeros(nu, dtype=np.float64)

    lx[:actual_state_dim] += (x[:actual_state_dim]-x_des).T @ Q
    lx[actual_state_dim : actual_state_dim + actual_input_dim] += -acc.T @ R_acc/dt - 2*jerk.T @ R_jerk/dt**2
    lx[actual_state_dim + actual_input_dim : actual_state_dim + 2*actual_input_dim] += jerk.T @ R_jerk/dt**2
    lx = lx*dt

    lu[:actual_input_dim] += (u-u_des).T @ R + acc.T @ R_acc/dt + jerk.T @ R_jerk/dt**2
    lu = lu*dt

    return lx, lu


"""ステージコストのヤコビアンを計算"""
@njit(cache=True,fastmath=True)
def l_jacb(
    #x: np.ndarray,
    #u: np.ndarray,
    #x_des: np.ndarray,
    #u_des: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    R_acc: np.ndarray,
    R_jerk: np.ndarray,
    dt: float = dt,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lxx = np.zeros((nx,nx), dtype=np.float64)
    luu = np.zeros((nu,nu), dtype=np.float64)
    lxu = np.zeros((nx,nu), dtype=np.float64)

    lxx[:actual_state_dim, :actual_state_dim] += Q
    lxx[actual_state_dim : actual_state_dim + actual_input_dim, actual_state_dim : actual_state_dim + actual_input_dim] \
        += R_acc/dt**2 + 4*R_jerk/dt**4
    lxx[actual_state_dim + actual_input_dim : actual_state_dim + 2*actual_input_dim, actual_state_dim + actual_input_dim : actual_state_dim + 2*actual_input_dim] \
        += R_jerk/dt**4
    lxx = lxx*dt
    
    luu[:actual_input_dim, :actual_input_dim] += R + R_acc/dt**2 + R_jerk/dt**4
    luu = luu*dt

    lxu[actual_state_dim : actual_state_dim + actual_input_dim, :actual_input_dim] += -R_acc/dt**2 - 2*R_jerk/dt**4
    lxu[actual_state_dim + actual_state_dim : actual_state_dim + 2*actual_input_dim, :actual_input_dim] += R_jerk/dt**4
    lxu = lxu*dt

    return lxx, luu, lxu


"""リカッチ方程式を解く（Backward Pass）"""
@njit(cache=True,fastmath=True)
def backward_pass(
    x: np.ndarray,
    u: np.ndarray,
    x_des: np.ndarray,
    u_des: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    R_acc: np.ndarray,
    R_jerk: np.ndarray,
    #F_diff: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = mobile_diff,
    dt: float = dt,
) -> Tuple[np.ndarray, np.ndarray, float, float] :
    N = x.shape[0]-1 # xはN+1個あるので

    Vx = np.zeros((nx,), dtype=np.float64)
    Vxx = np.zeros((nx,nx), dtype=np.float64)

    del_V1 = 0
    del_V2 = 0

    K = np.zeros((N,nu,nx), dtype=np.float64)
    d = np.zeros((N,nu), dtype=np.float64)

    for k in range(N-1, -1, -1):
        x_k = x[k]
        u_k = u[k]
        x_des_k = x_des[k]
        u_des_k = u_des[k]
        Q_k = Q[k]
        R_k = R[k]
        R_acc_k = R_acc[k]
        R_jerk_k = R_jerk[k]

        lxx, luu, lxu = l_jacb(Q_k, R_k, R_acc_k, R_jerk_k, dt)
        lux = lxu.T
        lx, lu = l_grad(x_k, u_k, x_des_k, u_des_k, Q_k, R_k, R_acc_k, R_jerk_k, dt)
        Ak, Bk = F_diff(x_k, u_k, dt)

        Qxx = lxx + Ak.T @ Vxx @ Ak
        Quu = luu + Bk.T @ Vxx @ Bk
        Qux = lux + Bk.T @ Vxx @ Ak

        Qx = lx + Vx @ Ak
        Qu = lu + Vx @ Bk

        try:
            kekka = np.linalg.cholesky(Quu)
        except:
            #もし違ったら
            #正定化の為にまず固有値の最小値を特定する
            print("seikika")
            eigenvalues, _ = np.linalg.eig(Quu)  # 固有値と固有ベクトルを分離
            alpa = -np.amin(eigenvalues)  # 固有値の最小値を取得して負号を付ける
            Quu = Quu + (alpa + 1e-2) * np.eye(nu) #正定化

        Quu_inv = np.linalg.inv(Quu)

        K_ = - Quu_inv @ Qux
        d_ = - Quu_inv @ Qu

        K[k] = K_
        d[k] = d_

        Vx = Qx - Qu @ Quu_inv @ Qux
        Vxx = Qxx - Qux.T @ Quu_inv @ Qux

        del_V1 += Qu @ d_
        del_V2 += 0.5 * d_ @ Quu @ d_

    del_V1 = float(del_V1)
    del_V2 = float(del_V2)

    return K, d, del_V1, del_V2


"""Forward Pass内でのロールアウト関数（微小範囲内でロールアウト+コストの計算）"""
@njit(cache=True,fastmath=True)
def rollout_check(
    x: np.ndarray,
    u: np.ndarray,
    x_des: np.ndarray,
    u_des: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    R_acc: np.ndarray,
    R_jerk: np.ndarray,
    K: np.ndarray,
    d: np.ndarray,
    alpha: float,
    #F_diff: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = mobile_diff,
    dt: float = dt,
) -> Tuple[np.ndarray, np.ndarray, float]:
    N = x.shape[0] - 1

    x_new = np.zeros(x.shape, dtype=np.float64)
    u_new = np.zeros(u.shape, dtype=np.float64)
    x_new[0] = x[0].copy()

    del_x = np.zeros(nx, dtype=np.float64)
    del_u = np.zeros(nu, dtype=np.float64)
    cost = 0

    for k in range(N):
        Ak, Bk = F_diff(x[k],u[k],dt)
        del_u = K[k] @ del_x + alpha*d[k]
        del_x = Ak @ del_x + Bk @ del_u
        u_new[k] = u[k] + del_u
        x_new[k+1] = x[k+1] + del_x
        cost += l_cost(x_new[k], u_new[k], x_des[k], u_des[k], Q[k] ,R[k], R_acc[k], R_jerk[k], dt)
    cost = float(cost)

    return x_new, u_new, cost


"""全コストを計算する関数"""
@njit(cache=True,fastmath=True)
def cost_calc(
    x: np.ndarray,
    u: np.ndarray,
    x_des: np.ndarray,
    u_des: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    R_acc: np.ndarray,
    R_jerk: np.ndarray,
    dt: float = dt,  
) -> float:
    N = x.shape[0] - 1
    cost = 0

    for k in range(N):
        cost += l_cost(x[k], u[k], x_des[k], u_des[k], Q[k] ,R[k], R_acc[k], R_jerk[k], dt)
    cost = float(cost)

    return cost


"""Forward Passの計算"""
@njit(cache=True,fastmath=True)
def forward_pass(
    x: np.ndarray,
    u: np.ndarray,
    x_des: np.ndarray,
    u_des: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    R_acc: np.ndarray,
    R_jerk: np.ndarray,
    K: np.ndarray,
    d: np.ndarray,
    del_V1: float,
    del_V2: float,
    #F_diff: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = mobile_diff,
    dt: float = dt,
) -> Tuple[np.ndarray, np.ndarray, float]:
    alpha = 1.0
    iter = 10

    cost_old = cost_calc(x,u,x_des,u_des,Q,R,R_acc,R_jerk,dt)

    for k in range(iter):
        x_new, u_new, cost = rollout_check(x, u, x_des, u_des, Q, R, R_acc, R_jerk, K, d, alpha, dt)
        del_V = alpha*del_V1 + (alpha**2)*del_V2
        z = (cost_old-cost)/(-del_V)
        alpha = 0.9 * alpha

        if 1e-4 <= z and z<= 10:
            break
    cost = float(cost)

    return x_new, u_new, cost


"""現在座標と現在入力から計算する関数"""
@njit(cache=True,fastmath=True)
def rollout(
    x_current: np.ndarray,
    u: np.ndarray,
    #F: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = mobile_F,
    dt: float = dt,
) -> np.ndarray:
    N = u.shape[0]
    x_ = x_current
    x_pred = np.zeros((N+1,nx), dtype=np.float64)
    x_pred[0] = x_

    for k in range(N):
        x_ = F(x_,u[k],dt)
        x_pred[k+1] = x_

    return x_pred



if __name__ == "__main__":
    pass