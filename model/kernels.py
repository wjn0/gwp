import numpy as np

def ou(t1, t2, params):
    """
    Ornstein-Uhlenbeck kernel. Commonly used for financial data because
    it's not quite as smooth as the squared-exponential kernel.
    """
    tau = params[0]
    
    return np.exp(-abs(t2 - t1)/tau)

def squared_exponential(t1, t2, params):
    """
    Squared-exponential kernel.
    """
    tau = params[0]
    
    return np.exp(-0.5*((t1 - t2)/tau)**2)

def periodic(t1, t2, params):
    """
    A simple periodic kernel function.
    """
    p, tau = params
    
    return np.exp(-2*np.sin(np.pi*(t1 - t2)/p)**2/tau**2)

def generate_sum_kernel(k1, k2, num_taus=(1, 1)):
    def k(t1, t2, params):
        p1, p2 = params[:num_taus[0]], params[num_taus[0]:]

        return 0.5 * k1(t1, t2, p1) + 0.5 * k2(t1, t2, p2)
    
    return k

def generate_prod_kernel(k1, k2, num_taus=(1, 1)):
    def k(t1, t2, params):
        p1, p2 = params[:num_taus[0]], params[num_taus[0]:]

        return k1(t1, t2, p1) * k2(t1, t2, p2)

    return k

