import numpy as np

def gbm_simulation(starting_point : float, drift_coefficient : float, 
                   time_steps : int, total_time : float, variance : float, num_sims : int,
                   random_draws : np.array = None) -> np.array:
    """Simulates a Geometric Brownian motion walk
    
    Parameters
    ----------
        starting_point : (float)

        drift_coefficient : (float) 
    
        time_steps : (int)

        total_time : (float)

        variance : (float)

        num_sim : (int) number of simulations to run 

        random_draws : (np.array) pre-sample random draws of size (time_steps, num_sims)

    Returns
    -------
    """
    # each time step 

    dt = total_time / time_steps

    if random_draws is None: random_draws = np.random.normal(0, 1, size=(time_steps, num_sims ))

    St = np.log(starting_point) + np.cumsum(
        (drift_coefficient - variance ** 2 / 2) * dt 
        + variance * np.sqrt(dt) * random_draws, axis=0)

    return np.exp(St)


if __name__ == '__main__':

    sims = gbm_simulation(
        100, 0.1, 100, 1, 0.2, 1000)
    
    print(len(sims[-1]))