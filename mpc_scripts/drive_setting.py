"""
汎用変数
"""
actual_state_dim = 3
actual_input_dim = 2
nx = actual_state_dim + 2*actual_input_dim
nu = actual_input_dim
N = 10
dt = 0.1
Th = N*dt

"""
MPPI用変数
"""
lam = 1
sample_num = 1000
mppi_tol = None