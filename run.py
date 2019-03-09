import numpy as np
from model.backtest import backtest_gp, backtest_gwp
from model.kernels import squared_exponential, periodic

data = np.loadtxt("data.txt")

print(backtest_gwp(data[:, :10], 8, squared_exponential))

