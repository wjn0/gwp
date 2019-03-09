from gwp import GeneralizedWishartProcess
from gp import GaussianProcess
from kernels import squared_exponential, periodic

class PortfolioChooser(object):
    def __init__(self, data, gwp_sig_var=0.95, gwp_kernel=squared_exponential,
                 gwp_tau_prior_mean=0.5, gwp_tau_prior_var=2,
                 gwp_L_prior_var=1e-2, gp_kernel=periodic, gp_nwalkers=100,
                 gp_tau_prior_mean=0.75, gp_tau_prior_var=1,
                 gp_sig_var_prior_alpha=10, gp_sig_var_beta=1.1,
                 gibbs_numit=1_000, mcmc_numit=500):
        self.data = data
        self.N, self.T = data.shape
        self.gwp_params = {
            'sig_var': gwp_sig_var,
            'kernel': gwp_kernel,
            'tau_prior_mean': gwp_tau_prior_mean,
            'tau_prior_var': gwp_tau_prior_var,
            'L_prior_var': gwp_L_prior_var,
        }
        self.gp_params = {
            'kernel': gp_kernel,
            'nwalkers': gp_nwalkers,
            'tau_prior_mean': gp_tau_prior_mean,
            'tau_prior_var': gp_tau_prior_var,
            'sig_var_prior_alpha': gp_sig_var_prior_alpha,
            'sig_var_prior_beta': gp_sig_var_prior_beta
        }
        self.gibbs_numit = 1_000
        self.mcmc_numit = 500

        self.gwp = GeneralizedWishartProcess(**self.gwp_params)
        self.gps = [
            GaussianProcess(**self.gp_params) for _ in range(data.shape[0])
        ]

    def fit_models(self):
        self.gwp_samples, self.gwp_diagnostics = self.gwp.fit(data, numit=self.gibbs_numit)
        self.gp_samples = [
            self.gps[i].fit(data[i, :], numit=self.mcmc_numit) for i in range(self.N)
        ]

    def compute(self):
        def neg_sharpe_ratio(w):
            return -np.dot(w, r) / np.sqrt(np.matmul(w, np.matmul(cov, w)))

        rpred = np.asarray([
            self.gps[i].predict_next_timepoint(self.data[i, :]) for i in range(self.N)
        ])
        sigpred = self.gwp.predict_next_timepoint(self.data)

        w0 = np.random.normal(size=len(r))
        w0 = w0 / np.linalg.norm(w0)
        res = minimize(neg_sharpe_ratio, w0)

        return res.x

