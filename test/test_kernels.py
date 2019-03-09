import sys
sys.path.insert(0, "/root/Projects/cryptoanalysis")

from model import kernels as ker

kernels = [ker.squared_exponential, ker.periodic, ker.ou]

def test_validity():
    test_scales = [0.01, 0.1, 1.0, 1.5, 2.0, 5.0, 10.0, 100.0]
    t_max = 100

    for k in kernels:
        for scale in test_scales:
            for t1 in range(t_max):
                assert(abs(k(0, t1, [scale])) <= 1)

