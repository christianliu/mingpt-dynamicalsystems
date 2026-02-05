import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = [8, 8]
# plt.rcParams.update({'font.size': 18})

def delayed_logistic(delay: int, r: float, x_init: list[float], n: int):
    """
    List[Float] of length delay + 1 (initial conditions) -> np.array of shape (n,)
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
    List[Dict{delay:, r:, x_init:}] -> List[np.array of shape (n,)]
    """
    num_traj = len(delayed_logistic_params)
    trajs = []
    for i, param in enumerate(delayed_logistic_params):
        trajs.append(delayed_logistic(param["delay"], param["r"], param["x_init"], n))
    return trajs

def plot_time_series(trajs, labels, separate_subplots = False):
    """
    List[np.array of shape (n,)] -> plot
    Plots trajectories across time
    """
    num_traj = len(trajs)
    n = trajs[0].shape[0] # assume all traj same length

    if separate_subplots:
        fig, axs = plt.subplots(num_traj, 1)
        if num_traj == 1: # Make axs iterable even if only one trajectory
            axs = [axs]
        for i, ax in enumerate(axs):
            # ax.scatter(np.arange(min(500, n)), trajs[i][-500:], color='k', s = 5) # plot last min(n, 500) points
            ax.plot(np.arange(min(500, n)), trajs[i][-500:], color='k') # plot last min(n, 500) points
            ax.set_title(labels[i])
            ax.set_xlabel("n")
            ax.set_ylabel("x_n")
        plt.tight_layout()
        plt.show()
    else:
        fig = plt.figure()
        for i in range(num_traj):
            # plt.scatter(np.arange(min(500, n)), trajs[i][-500:], s = 5, label=labels[i])
            plt.plot(np.arange(min(500, n)), trajs[i][-500:], label=labels[i])
        plt.xlabel("n")
        plt.ylabel("x_n")
        plt.title("Delayed logistic map trajectories")
        plt.legend()
        plt.show()
    return fig

def plot_phase_space(trajs, labels, shifts, separate_subplots = True):
    """
    List[Dict{delay:, r:, x_init:}] -> plot
    Plots x_{n-shift} vs. x_n as approx to phase space (inspired by Taken's theorem)
    shifts is a list of ints, can choose a different shift for each traj
    Want two var to not be too correlated (no info if shift too small), nor completely uncorrelated (shift too large)
    Pick where autocorrelation first crosses 1/e or drops significantly
    """
    num_traj = len(trajs)
    n = trajs[0].shape[0] # assume all traj same length

    if separate_subplots:
        fig, axs = plt.subplots(num_traj, 1, figsize=(6, 4*num_traj))
        if num_traj == 1:
            axs = [axs]
        for i, ax in enumerate(axs):
            shift = shifts[i]
            ax.plot(trajs[i][-500-shift:-shift], trajs[i][-min(500, n-shift):], '.', markersize=1) # plot last min(n-shift, 500) points
            ax.set_xlabel(f"x_(n-{shift})")
            ax.set_ylabel("x_n")
            ax.set_title(labels[i])
        plt.tight_layout()
        plt.show()
    else:
        fig = plt.figure(figsize=(6,6))
        for i in range(num_traj):
            shift = shifts[i]
            plt.plot(trajs[i][-500-shift:-shift], trajs[i][-min(500, n-shift):], '.', markersize=1, label=labels[i])
        plt.xlabel(f"x_(n-shift)")
        plt.ylabel("x_n")
        plt.title("Phase-space reconstruction of delayed logistic map")
        plt.legend()
        plt.show()
    return fig

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
    trajs = delayed_logistic_mult(delayed_logistic_params, n)
    print(trajs[0])
    labels = [f"r={param['r']}, delay={param['delay']}, x0={param['x_init']}" for param in delayed_logistic_params]
    shifts = [param["delay"] for param in delayed_logistic_params]

    plot_time_series(trajs, labels, False)
    plot_phase_space(trajs, labels, shifts, True)

