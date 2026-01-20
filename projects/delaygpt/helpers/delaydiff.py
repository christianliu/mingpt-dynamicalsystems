import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = [8, 8]
# plt.rcParams.update({'font.size': 18})
separate_subplots = True

def delayed_logistic(delay: int, r: float, x_init: list[float], n: int):
    """
    Returns length n sequence by generating element i from elements i-1 and i-1-delay and param r: 
    x_{i+1} = r * x_i * (1-x_{i-delay})
    Need delay+1 many initial conditions
    Smaller delay less chaotic, larger delay goes to infinity
    """
    delay = int(delay)
    if len(x_init) != delay + 1:
        raise ValueError(f"x_init length {len(x_init)} must equal delay+1 = {delay+1}")
    if n <= len(x_init):
        return x_init[:n]
    
    x = np.zeros(n)
    x[: delay+1] = x_init
    for i in range(delay, n - 1):
        x[i + 1] = r * x[i] * (1 - x[i - delay])
    return x

def delayed_logistic_mult(delayed_logistic_params, n: int):
    """
    Applies delayed_logistic to several sets of params at once, returns trajectories in (num_traj, n) array
    """
    num_traj = len(delayed_logistic_params)
    xs = np.zeros((num_traj, n))
    for i, par in enumerate(delayed_logistic_params):
        xs[i] = delayed_logistic(par["delay"], par["r"], par["x_init"], n)
    return xs

def plot_time_series(xs, delayed_logistic_params, separate_subplots = False):
    """
    Plots trajectories across time
    """
    if separate_subplots:
        fig, axs = plt.subplots(len(xs), 1)
        if len(xs) == 1:         # Make axs iterable even if only one trajectory
            axs = [axs]
        for i, ax in enumerate(axs):
            ax.scatter(np.arange(500), xs[i, -500:], color='k', s = 5)
            par = delayed_logistic_params[i]
            ax.set_title(f"r={par['r']}, delay={par['delay']}, x0={par['x_init']}")
            ax.set_xlabel("n")
            ax.set_ylabel("x_n")
        plt.tight_layout()
        plt.show()
    else:
        plt.figure()
        for i, par in enumerate(delayed_logistic_params):
            plt.scatter(np.arange(500), xs[i, -500:], s = 5,
                        label=f"r={par['r']}, delay={par['delay']}, x0={par['x_init']}")
        plt.xlabel("n")
        plt.ylabel("x_n")
        plt.title("Delayed logistic map trajectories")
        plt.legend()
        plt.show()

def plot_phase_space(xs, delayed_logistic_params, separate_subplots = True):
    """
    Plot x_{n-t} vs. x_n as approx to phase space (inspired by Taken's theorem)
    Want two var to not be too correlated (no info if t too small), nor completely uncorrelated (t too large)
    Pick where autocorrelation first crosses 1/e or drops significantly
    WARNING: t chosen automatically as delay param here, maybe add functionality to choose later
    """
    if separate_subplots:
        fig, axs = plt.subplots(len(xs), 1, figsize=(6, 4*len(xs)))
        if len(xs) == 1:
            axs = [axs]
        for i, ax in enumerate(axs):
            par = delayed_logistic_params[i]
            delay = int(par["delay"])
            ax.plot(xs[i,-5000-delay:-delay], xs[i,-5000:], '.', markersize=1)
            ax.set_xlabel(f"x_(n-{delay})")
            ax.set_ylabel("x_n")
            ax.set_title(f"Phase-space: r={par['r']}, delay={delay}, x0={par['x_init']}")
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(6,6))
        for i, par in enumerate(delayed_logistic_params):
            delay = int(par["delay"])
            plt.plot(xs[i, -5000-delay:-delay], xs[i, -5000:], '.', markersize=1, label=f"r={par['r']}, delay={delay}, x0={par['x_init']}")
        plt.xlabel(f"x_(n-delay)")
        plt.ylabel("x_n")
        plt.title("Phase-space reconstruction of delayed logistic map")
        plt.legend()
        plt.show()


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
    xs = delayed_logistic_mult(delayed_logistic_params, n)
    # plot_time_series(xs, delayed_logistic_params, False)
    # plot_phase_space(xs, delayed_logistic_params, True)

    print(xs[0])
    print(xs[2])

