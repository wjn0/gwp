import numpy as np
import emcee

class GeneralizedWishartProcess(object):
    def __init__(self, sig_var, kernel, tau_prior_mean, tau_prior_var,
                 L_prior_var):
        """
        Initialize the model with parameters.

        sig_var: The signal variance for the kernel. Used to keep it PSD.
        kernel:  The kernel function to use. Must accept arguments like
                 (t1, t2, tau).
        tau_prior_*: The prior over tau will be
                     LogNormal(tau_prior_mean, tau_prior_var).
        L_prior_var: The prior (Gaussian) variance over the elements of L.
                     Should be set to the same order of magnitude of the
                     elements of the dataset.
        """
        self.parameters = {
            'SIG_VAR': sig_var,
            'KERNEL': kernel,
            'TAU_PRIOR_MEAN': tau_prior_mean,
            'TAU_PRIOR_VAR': tau_prior_var,
            'L_PRIOR_VAR': L_prior_var,
            'MH_L_SCALE': L_prior_var/10
        }

    def _construct_kernel(self, params, times):
        """
        Construct the kernel for the GPs which generate the GWP.

        params: The kernel parameters of dimension ν x N x h where ν is the
                d.f., N is the dimensionality of the data, and h is the number
                of parameters for the kernel function.
        times:  The timepoints to generate for.
        
        Returns a νNT x νNT kernel matrix.
        """
        def kidx(n, t):
            """
            Generates a lambda which maps indices of u to their flattened
            positions.
            """
            return lambda a, b, c: a * (n * t) + b * n + c
        T = len(times)
        Nu, N, h = params.shape
        kernel_idx = kidx(N, T)

        K = np.eye(np.prod([Nu, N, T]))
        k = self.parameters['KERNEL']
        for nu in range(Nu):
            for n in range(N):
                for t1 in range(T):
                    for t2 in range(t1, T):
                        if t1 != t2:
                            i = kernel_idx(nu, n, t1)
                            j = kernel_idx(nu, n, t2)
                            K[i, j] = k(times[t1], times[t2], [params[nu, n, 0],
                                                               self.parameters['SIG_VAR']])

        return K

    def compute_sigma(self, L, u):
        """
        Compute the covariance matrix for a specific timepoint.

        L: The lower cholesky decomposition of the scale parameter for the
           Wishart distribution (of dimension N x N).
        u: The fitted GP function values that generate the draw from the Wishart
           distribution. Dimensionality: ν x N.

        Returns the N x N covariance matrix.
        """
        Nu = u.shape[0]
        Sig = np.zeros(L.shape)
        for nu in range(Nu):
            Sig += np.matmul(L, np.matmul(np.outer(u[nu, :], u[nu, :]), L.T))

        return Sig

    def _log_data_likelihood(self, data, u, L, Nu):
        """
        We use the simplest possible data likelihood: sum over all times t in
        [T], computing the probability of observing the data given that it comes
        from the distribution r(t) ~ N(0, Σ(t)).

        data: The observed data, of shape N x T.
        u:    The flattened vector of constructive GP function values.
        L:    The lower cholesky decomposition of the scale Wishart prior.
        Nu:   d.f.

        Returns the log-likelihood of observing the data given the model
        parameters.
        """
        loglik = 0
        N, T = data.shape
        u = np.reshape(u, (Nu, N, T))
        for t in range(T):
            Siginv = np.linalg.inv(self.compute_sigma(L, u[:, :, t]))
            term = -0.5*np.matmul(data[:, t].T, np.matmul(Siginv, data[:, t]))
            loglik += term

        return loglik

    def _sample_u(self, f, tau, T, L, Nu, data):
        """
        Sample u (equation 15). We use elliptical slice sampling, specifically a
        direct implementation of the algorithm in figure 2 from the original ESS
        paper.
        """
        K = self._construct_kernel(tau, range(T))
        Kinv = np.linalg.inv(K)

        ellipse = np.random.multivariate_normal(np.zeros(K.shape[0]), K)
        u = np.random.uniform()
        logy = self._log_data_likelihood(data, f, L, Nu) + np.log(u)
        angle = np.random.uniform(high=2*np.pi)
        angle_min, angle_max = angle - 2*np.pi, angle
        while True:
            fp = f*np.cos(angle) + ellipse*np.sin(angle)
            log_data_lik = self._log_data_likelihood(data, fp, L, Nu)
            if log_data_lik > logy:
                log_u_lik = -0.5*np.matmul(fp, np.matmul(Kinv, fp))
                return fp, log_data_lik + log_u_lik
            else:
                if angle < 0:
                    angle_min = angle
                else:
                    angle_max = angle
                angle = np.random.uniform(angle_min, angle_max)

    def _sample_logtau(self, logtau, u, T, L, Nu, data):
        """
        Sample tau (equation 16). We use standard Metropolis-Hastings, as
        implemented in the emcee library, to sample the next position in the
        chain.
        """
        def log_logtau_prob(logtaup):
            logtaup = np.reshape(logtaup, logtau.shape)
            K = self._construct_kernel(np.exp(logtaup), range(T))
            Kinv = np.linalg.inv(K)
            log_u_prob = -0.5*np.matmul(u, np.matmul(Kinv, u))
            mean = self.parameters['TAU_PRIOR_MEAN']
            var = self.parameters['TAU_PRIOR_VAR']
            log_prior = np.sum(-0.5*((logtaup - mean)**2/var))

            return log_u_prob + log_prior

        dim = np.prod(logtau.shape)
        sampler = emcee.MHSampler(np.eye(dim), dim=dim,
                                  lnprobfn=log_logtau_prob)
        logtaup, _, _ = sampler.run_mcmc(logtau.flatten(), 1)

        return np.reshape(logtaup, logtau.shape), log_logtau_prob(logtaup)

    def _sample_L(self, L, tau, u, Nu, data):
        """
        Sample L (equation 17). We use standard Metropolis-Hastings, as
        implemented in the emcee library, to sample the next position in the
        chain.
        """
        def log_L_prob(Lp):
            Lpm = np.zeros(L.shape)
            Lpm[np.tril_indices(L.shape[0])] = Lp
            log_prior = np.sum(-0.5 * Lp**2 / self.parameters['L_PRIOR_VAR'])

            return self._log_data_likelihood(data, u, Lpm, Nu) + log_prior

        dim = int((L.shape[0]**2 + L.shape[0])/2)
        scale = self.parameters['MH_L_SCALE']
        sampler = emcee.MHSampler(np.eye(dim) * scale, dim=dim,
                                  lnprobfn=log_L_prob)
        Lp, _, _ = sampler.run_mcmc(L[np.tril_indices(L.shape[0])], 1)
        Lpm = np.zeros(L.shape)
        Lpm[np.tril_indices(L.shape[0])] = Lp
        
        return Lpm, log_L_prob(Lp)

    def _init_u(self, T, tau):
        N, Nu, _ = tau.shape
        K = self._construct_kernel(tau, range(T))
        draw = np.random.multivariate_normal(np.zeros(K.shape[0]), K)

        return draw

    def _init_logtau(self, Nu, N):
        return np.random.normal(size=(Nu, N, 1))

    def _init_L(self, N):
        L = np.eye(N)

        return L * self.parameters['L_PRIOR_VAR']

    def fit(self, data, init=None, numit=1_000, progress=10):
        """
        Fit the model using a Gibbs sampling routine.

        data: The data to fit on. Dimension N x T, where N is the number of
              assets and T is the number of timepoints. Element (n, t) is the
              return of the nth asset at time t.

        Returns the chain of samples and diagnostics (likelihood and
        posterior probabilities).
        """
        samples, diagnostics = [], []

        N, T = data.shape
        Nu = N + 1
        self.Nu, self.N = Nu, N

        if init:
            logtau = init['logtau']
            u = init['u']
            L = init['L']
        else:
            u = self._init_u(T, np.exp(self._init_logtau(Nu, N)))
            logtau = self._init_logtau(Nu, N)
            L = self._init_L(N)

        samples.append([u, np.exp(logtau), L])

        for it in range(numit):
            data_lik = self._log_data_likelihood(data, u, L, Nu)
            u, u_prob = self._sample_u(u, np.exp(logtau), T, L, Nu, data)
            logtau, logtau_prob = self._sample_logtau(logtau, u, T, L, Nu, data)
            L, L_prob = self._sample_L(L, np.exp(logtau), u, Nu, data)
            
            samples.append([u, np.exp(logtau), L])
            diagnostics.append([data_lik, u_prob, logtau_prob, L_prob])

            if progress and it % progress is 0:
                if it >= numit // 5:
                    print(
                        "Best ({}): loglik = {:.2f}, log P(u|...) = {:.2f}, log P(tau|...) = {:.2f}, log P(L|...) = {:.2f}".format(
                            it, *max(diagnostics[(numit // 5):], key=lambda a: a[0])
                        )
                    )
                else:
                    print(
                        "Iter {}: loglik = {:.2f}, log P(u|...) = {:.2f}, log P(tau|...) = {:.2f}    , log P(L|...) = {:.2f}".format(
235                             it, *diagnostics[-1]
                        )
                    )
            
        self.samples = samples
        self.diagnostics = np.asarray(diagnostics)

        return samples, diagnostics

    def optimal_params(self, burnin=200):
        return self.samples[np.argmax(self.diagnostics[burnin:, 0]) + burnin]

    def _predict_next_u(self, T, burnin=200):
        u, tau, L = self.optimal_params(burnin)

        K = self._construct_kernel(tau, range(T + 1))
        idxs = np.full(K.shape[0], True)
        idxs[np.asarray(list(range(T, int(len(u)/T*(T+1)), T + 1)))] = False
        Kbinv = np.linalg.inv(K[idxs, :][:, idxs])
        A = K[np.logical_not(idxs), :][:, idxs]
        ustar = np.matmul(np.matmul(A, Kbinv), u)
        
        return ustar

    def predict_next_timepoint(self, data, burnin=200):
        u, tau, L = self.optimal_params(burnin)
        ustar = self._predict_next_u(data.shape[1], burnin)
        ustar = np.reshape(ustar, (self.Nu, self.N, 1))

        return self.compute_sigma(L, ustar[:, :, 0])

