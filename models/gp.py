import numpy as np
import emcee

class GaussianProcess(object):
    def __init__(self, kernel, nwalkers, tau_prior_mean, tau_prior_var,
                 sig_var_prior_alpha, sig_var_prior_beta):
        self.parameters = {
            'KERNEL': kernel,
            'NWALKERS': nwalkers,
            'TAU_PRIOR_MEAN': tau_prior_mean,
            'TAU_PRIOR_VAR': tau_prior_var,
            'SIG_VAR_PRIOR_ALPHA': sig_var_prior_alpha,
            'SIG_VAR_PRIOR_BETA': sig_var_prior_beta
	}

    def _construct_kernel(self, tau, sig_var, times):
        k = self.parameters['KERNEL']
        T = len(times)
        K = np.eye(T)
        for t1 in range(T):
            for t2 in range(t1, T):
                if t1 != t2:
                    el = sig_var * k(times[t1], times[t2], [tau])
                    K[t1, t2] = el
                    K[t2, t1] = el

        return K

    def fit(self, data, numit=1_000):
        tau_mean = self.parameters['TAU_PRIOR_MEAN']
        tau_var = self.parameters['TAU_PRIOR_VAR']
        sig_alpha = self.parameters['SIG_VAR_PRIOR_ALPHA']
        sig_beta = self.parameters['SIG_VAR_PRIOR_BETA']

        def log_likelihood(params):
            tau, sig_var = params
            if tau > 0 and sig_var < 1 and sig_var > 0:
                K = self._construct_kernel(tau, sig_var, range(len(data)))
                Kinv = np.linalg.inv(K)

                loglik = -0.5 * np.matmul(data, np.matmul(Kinv, data))
                logtauprior = np.log(1/tau) - 0.5*(np.log(tau) - tau_mean)**2/tau_var
                logsigprior = np.log(sig_var**(sig_alpha - 1)) + np.log((1 - sig_var)**(sig_beta - 1))

                return loglik + logtauprior + logsigprior
            else:
                return -np.inf

        nwalkers = self.parameters['NWALKERS']
        init = [
            [np.random.lognormal(tau_mean, tau_var),
             np.random.beta(sig_alpha, sig_beta)] for _ in range(nwalkers)
        ]

        sampler = emcee.EnsembleSampler(nwalkers, 2, log_likelihood)
        ret = sampler.run_mcmc(init, numit)
        self.chain = sampler.chain
        self.terminals = ret[0]
        self.likelihoods = ret[1]
         
        return self.chain

    def optimal_params(self, burnin=50):
        return self.terminals[np.argmax(self.likelihoods[burnin:]) + burnin, :]

    def predict_next_timepoint(self, data, burnin=50):
        tau, sig_var = self.optimal_params(burnin)
        T = len(data)
        K = self._construct_kernel(tau, sig_var, range(T + 1))

        return np.matmul(K[:T, T], np.matmul(np.linalg.inv(K[:T, :T]), data))

