import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = [8, 8]
# plt.rcParams.update({'font.size': 18})

def delayed_logistic(delay: int, r: float, x_init: list[float], n: int):
    """
    List[Float] of length delay + 1 (initial conditions) -> List[Float] of length n (path)
    x_{i+1} = r * x_i * (1-x_{i-delay})
    Smaller delay less chaotic, larger delay goes to infinity
    """
    delay = int(delay)
    if len(x_init) != delay + 1:
        raise ValueError(f"x_init length {len(x_init)} must equal delay+1 = {delay+1}")
    if n <= len(x_init):
        return x_init[:n]
    
    x = np.zeros(n)
    x[:delay+1] = x_init
    for i in range(delay, n-1): # for i = delay, ..., n-2, produce next value
        x[i+1] = r * x[i] * (1 - x[i-delay])
    return x

def delayed_logistic_mult(delayed_logistic_params, n: int):
    """
    List[Dict{delay:, r:, x_init:}] -> np.array of size (num_traj, n)
    """
    num_traj = len(delayed_logistic_params)
    trajs = np.zeros((num_traj, n))
    for i, param in enumerate(delayed_logistic_params):
        trajs[i] = delayed_logistic(param["delay"], param["r"], param["x_init"], n)
    return trajs

def plot_time_series(delayed_logistic_params, n, separate_subplots = False):
    """
    List[Dict{delay:, r:, x_init:}] -> plot
    Plots trajectories across time
    """
    num_traj = len(delayed_logistic_params)
    trajs = delayed_logistic_mult(delayed_logistic_params, n)

    if separate_subplots:
        fig, axs = plt.subplots(num_traj, 1)
        if num_traj == 1: # Make axs iterable even if only one trajectory
            axs = [axs]
        for i, ax in enumerate(axs):
            ax.scatter(np.arange(min(500, n)), trajs[i, -500:], color='k', s = 5) # plot last min(n, 500) points
            param = delayed_logistic_params[i]
            ax.set_title(f"r={param['r']}, delay={param['delay']}, x0={param['x_init']}")
            ax.set_xlabel("n")
            ax.set_ylabel("x_n")
        plt.tight_layout()
        plt.show()
    else:
        plt.figure()
        for i, param in enumerate(delayed_logistic_params):
            plt.scatter(np.arange(min(500, n)), trajs[i, -500:], s = 5,
                        label=f"r={param['r']}, delay={param['delay']}, x0={param['x_init']}")
        plt.xlabel("n")
        plt.ylabel("x_n")
        plt.title("Delayed logistic map trajectories")
        plt.legend()
        plt.show()
    return trajs

def plot_phase_space(delayed_logistic_params, n, separate_subplots = True):
    """
    List[Dict{delay:, r:, x_init:}] -> plot
    Plots x_{n-t} vs. x_n as approx to phase space (inspired by Taken's theorem)
    WARNING: t chosen automatically as delay param here, maybe add functionality to choose later
    Want two var to not be too correlated (no info if t too small), nor completely uncorrelated (t too large)
    Pick where autocorrelation first crosses 1/e or drops significantly
    """
    num_traj = len(delayed_logistic_params)
    trajs = delayed_logistic_mult(delayed_logistic_params, n)

    if separate_subplots:
        fig, axs = plt.subplots(num_traj, 1, figsize=(6, 4*num_traj))
        if num_traj == 1:
            axs = [axs]
        for i, ax in enumerate(axs):
            param = delayed_logistic_params[i]
            delay = int(param["delay"])
            ax.plot(trajs[i,-500-delay:-delay], trajs[i,-min(500, n-delay):], '.', markersize=1) # plot last min(n-delay, 500) points
            ax.set_xlabel(f"x_(n-{delay})")
            ax.set_ylabel("x_n")
            ax.set_title(f"Phase-space: r={param['r']}, delay={delay}, x0={param['x_init']}")
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(6,6))
        for i, param in enumerate(delayed_logistic_params):
            delay = int(param["delay"])
            plt.plot(trajs[i,-500-delay:-delay], trajs[i,-min(500, n-delay):], '.', markersize=1, label=f"r={param['r']}, delay={delay}, x0={param['x_init']}")
        plt.xlabel(f"x_(n-delay)")
        plt.ylabel("x_n")
        plt.title("Phase-space reconstruction of delayed logistic map")
        plt.legend()
        plt.show()
    return trajs


if __name__ == '__main__':
    n = int(100)
    delayed_logistic_params = [
        {"r": 2.26, "delay": 1, "x_init": [0.1, 0.1]},
        {"r": 2.26, "delay": 1, "x_init": [0.1, 0.15]},
        {"r": 2.26, "delay": 1, "x_init": [0.1, 0.2]}
        # {"r": 2.2, "delay": 1, "x_init": [0.1, 0.1]},
        # {"r": 1.5, "delay": 1, "x_init": [0.1, 0.1]},
        # {"r": 1.5, "delay": 3, "x_init": [0.1, 0.1, .1, .1]}
    ]
    # trajs = delayed_logistic_mult(delayed_logistic_params, n)
    trajs = plot_time_series(delayed_logistic_params, n, False)
    trajs = plot_phase_space(delayed_logistic_params, n, True)

    print(trajs[0])

