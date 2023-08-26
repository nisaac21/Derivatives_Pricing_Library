import numpy as np

def gbm_simulation(starting_point : float, drift_coefficient : float, 
                   time_steps : int, total_time : float, variance : float, num_sims : int) -> np.array:
    """Simulates a Geometric Brownian motion walk
    
    Parameters
    ----------
        starting_point : (float)

        drift_coefficient : (float) 
    
        time_steps : (int)

        total_time : (float)

        variance : (float)

        num_sim : (int) number of simulations to run 
    Returns
    -------
    """
    # each time step 

    dt = total_time / time_steps

    St = np.log(starting_point) + np.cumsum(
        (drift_coefficient - variance ** 2 / 2) * dt 
        + variance * np.random.normal(0, np.sqrt(dt), size=(time_steps, num_sims )), axis=0)

    return np.exp(St)