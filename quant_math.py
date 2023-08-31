import numpy as np


def gbm_simulation(S0: float, mu: float,
                   n_steps: int, T: float, sigma: float, num_sims: int,
                   random_draws: np.array = None) -> np.array:
    """Simulates a Geometric Brownian motion walk

    Parameters
    ----------
        S0 : (float) starting point for randomness

        mu : (float) drift coefficient

        n_steps : (int) number of time steps to take 

        T : (float) total time to simulate over

        sigma : (float) variance

        num_sim : (int) number of simulations to run 

        random_draws : (np.array) pre-sample random draws of size (time_steps, num_sims)

    Returns
    -------
    """
    # each time step

    dt = T / n_steps

    if random_draws is None:
        random_draws = np.random.normal(0, 1, size=(n_steps, num_sims))

    St = np.log(S0) + np.cumsum(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.sqrt(dt) * random_draws, axis=0)

    return np.exp(np.vstack(np.log(S0), St))


def merton_jump_diff(S0: float, mu: float, sigma: float, T: float,  n_steps: int, num_sims: int,
                     lambda_j: float = 0.1, mu_j: float = -0.2,  sigma_j: float = 0.3,
                     random_draws: np.array = None, random_jump_draws: np.array = None):
    """ A jump-diffusion model for path simulation based off of Merton's analytical formula

    Parameters
    ----------
        S0 : (float) starting point for randomness

        mu : (float) drift coefficient

        sigma : (float) variance

        n_steps : (int) number of time steps to take 

        T : (float) total time to simulate over

        num_sim : (int) number of simulations to run 

        lambda_j : (float) jump intensity

        mu_j : (float) average jump size 

        sigma_j : (float) jump size volatility

        random_draws : (np.array) pre-sample random draws of size (time_steps, num_sims)

        random_jump_draws : (np.array) pre-sample random draws for the jump component of the model in size (time_steps, num_sims)

    Returns
    -------
    """

    dt = T / n_steps

    if random_draws is None:
        random_draws = np.random.normal(0, 1, size=(n_steps, num_sims))

    if random_jump_draws is None:
        random_jump_draws = np.random.normal(mu_j, sigma_j, size=(
            n_steps, num_sims)) * np.random.poisson(lambda_j * dt, size=(n_steps, num_sims))

    St = np.log(S0) + np.cumsum(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.sqrt(dt) * random_draws + random_jump_draws, axis=0)

    return np.exp(np.vstack(np.log(S0), St))


def heston_path(S0: float, mu: float, n_steps: int, T: float,
                sigma: float, corr: float, epsilon: float,
                kappa: float, theta: float, num_sims: int,
                random_draws: np.array = None) -> np.array:
    """Simulates a Geometric Brownian motion walk

    Parameters
    ----------
        S0 : (float) starting point for randomness

        mu : (float) drift coefficient

        T : (float) total time to simulate over

        sigma : (float) variance

        corr : (float) correlation between asset returns and variance 

        epsilon : (float) variance of volatility distribution

        kappa : (float) rate of mean reversion for volatility process

        theta : (float) long-term mean of variance process 

        n_steps : (int) number of time steps to take 

        num_sim : (int) number of simulations to run 

        random_draws : (np.array) pre-sample random draws which must be
                        np.random.multivariate_normal(mu=[0, 0], 
                                                      cov=np.array([[1, corr], [corr, 1]]),
                                                      size=(num_sims, n_steps))

    Returns
    -------
        path : (np.array) for asset prices over time 
    """
    # each time step

    dt = T / n_steps

    prices = np.full(shape=(num_sims, n_steps+1), fill_value=S0)
    volatility = np.full(shape=(num_sims, n_steps+1), fill_value=sigma)
    S_t = S0
    v_t = sigma

    cov_matrix = np.array([[1, corr], [corr, 1]])

    # random variable with relationship corr
    WT = np.random.multivariate_normal(
        mu=np.array([0, 0]),
        cov=cov_matrix,
        size=(num_sims, n_steps)) * np.sqrt(dt) if random_draws is None else random_draws

    for t in range(1, n_steps+1):

        S_t = S_t*(np.exp((mu - 0.5*v_t)*dt + np.sqrt(v_t) * WT[:, t-1, 0]))

        v_t = np.abs(v_t + kappa*(theta-v_t)*dt +
                     epsilon * np.sqrt(v_t)*WT[:, t-1, 1])

        prices[:, t] = S_t
        volatility[:, t] = v_t

    return prices
