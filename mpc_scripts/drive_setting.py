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