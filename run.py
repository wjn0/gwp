import numpy as np
from model.backtest import backtest_gp, backtest_gwp
from model.kernels import squared_exponential, periodic, generate_sum_kernel

data = np.loadtxt("data.txt")

print(backtest_gp(data[:2, :10], 8, generate_sum_kernel(periodic, squared_exponential),
                  num_taus=2, initial_numit=5_000))

