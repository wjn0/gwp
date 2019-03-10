import sys
import numpy as np

from model.gp import GaussianProcess as GP
from model.gwp import GeneralizedWishartProcess as GWP

def _extend_gwp_params(gwp, burnin, t):
    N, Nu = gwp.N, gwp.Nu
    ustar = gwp._predict_next_u(t, burnin)
    opt = gwp.optimal_params(burnin)
    uinit = np.concatenate((opt[0], ustar))

    init = {
        'logtau': np.log(opt[1]),
        'u': uinit,
        'L': opt[2]
    }

    return init

def backtest_gwp(full_data, start_point, gwp_kernel, initial_numit=1_000,
                 successive_numit=500):
    N, T = full_data.shape
    predictions = []

    print("Fitting initial model (T = {})...".format(start_point))

    gwp = GWP(
        0.95,
        gwp_kernel,
        0.5, 2,
        1e-1
    )
    gwp.fit(full_data[:, :start_point], numit=initial_numit)

    burnin = initial_numit // 5
    predictions.append(
        gwp.predict_next_timepoint(full_data[:, :start_point], burnin=burnin)
    )
    for t in range(start_point + 1, T):
        print("Fitting models for T = {}...".format(t))
        sys.stdout.flush()

        init = _extend_gwp_params(gwp, burnin, t - 1)
        gwp.fit(full_data[:, :t], init, numit=successive_numit)

        burnin = successive_numit // 5
        predictions.append(
            gwp.predict_next_timepoint(full_data[:, :t], burnin=burnin)
        )

    return predictions

def backtest_gp(full_data, start_point, gp_kernel, initial_numit=500,
                successive_numit=250, num_taus=1):
    """
    Perform backtesting on as much of the data as possible. This is equivalent
    to fitting GP models to D[t] and predicting r[t + 1]
    for t = start_point, ..., T where T is the number of timepoints in
    full_data. Here, D[t] is all the training data from time 0 to time t. This
    is much more efficient than doing so manually: we use the previous model's
    parameters to efficiently sample the parameters for the data including the
    next timepoint, therefore requiring much fewer iterations of
    the GP model. Convergence still has to be assessed manually.

    full_data: Matrix of size N x T, where N is the number of assets and T is
               the number of timepoints. Should be standardized.
    start_point: The timepoint to start predictions at (the first predictions
                 will be for r[start_point] and Σ[start_poit].

    Returns the predictions of r[t] and S[t] for t = start_point + 1, ..., T and
    the true, normalized values.
    """
    N, T = full_data.shape
    predictions = []

    # Fit the initial models.
    print("Fitting initial models (T = {})...".format(start_point), end=" ")
    sys.stdout.flush()
    gps = [
        GP(
            gp_kernel,
            100,
            0.75, 2,
            10, 1.1,
            num_taus=num_taus
        ) for _ in range(N)
    ]
    for n in range(N):
        gps[n].fit(full_data[n, :start_point], numit=initial_numit)

        print(n, end=" ")
        sys.stdout.flush()

    print("")
    sys.stdout.flush()

    print(np.asarray([
        gps[n].optimal_params() for n in range(N)
    ]))

    predictions.append([
        gps[n].predict_next_timepoint() for n in range(N)
    ])


    for t in range(start_point + 1, T):
        print("Fitting models for T = {}...".format(t))
        sys.stdout.flush()

        init = list(map(lambda gp: gp.terminals, gps))
        for n in range(N):
            gps[n].fit(full_data[n, :t], numit=successive_numit)

        predictions.append([
            gps[n].predict_next_timepoint() for n in range(N)
        ])

        print(np.asarray([
            gps[n].optimal_params() for n in range(N)
        ]))

    return predictions

