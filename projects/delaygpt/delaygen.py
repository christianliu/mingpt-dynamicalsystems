import os

import torch
import numpy as np

from mingpt.plot_att import load_gpt_from_dir, plot_att_from_input
from helpers.delaydiff import delayed_logistic_mult, plot_time_series, plot_phase_space

if __name__ == '__main__':
    # data the trained model was trained on
    # train_n = int(1e6)
    # train_params = [ # train on 2 paths of one equation
    #     {"r": 2.26, "delay": 1, "x_init": [0.1, 0.1]},
    #     {"r": 2.26, "delay": 1, "x_init": [0.1, 0.15]}
    # ]
    
    # get trained model
    work_dir = "out/delaygpt"
    delay_model = load_gpt_from_dir(work_dir, cts_model=True)
    block_size = delay_model.block_size

    # get traj of trained model and truth for 2 trajectories
    compare_n = int(150)
    test_params = [ # test on 2 paths of same equation
        {"r": 2.26, "delay": 1, "x_init": [0.1, 0.2]},
        {"r": 2.26, "delay": 1, "x_init": [0.1, 0.175]}
    ]
    # fn: List[Dict{delay:, r:, x_init:}] -> List[np.array of shape (n,)]
    true_trajs = delayed_logistic_mult(test_params, compare_n)
    # List[np.array of shape (block_size,)] -> tensor of shape (num_traj, block_size, 1)
    model_input = torch.from_numpy(np.stack([x[:block_size] for x in true_trajs]).astype(np.float32)).unsqueeze(2) # musst be type float, print(type(true_trajs[0][0]))
    model_output = delay_model.generate(model_input, compare_n - block_size).y
    # tensor of shape (num_traj, block_size, 1) -> List[List of len n] -> List[np.array of shape (n,)]
    model_trajs = [np.array(x) for x in model_output.squeeze(2).tolist()]
    np.savez(os.path.join(work_dir, "model_output.npz"), model_trajs)

    # plot
    trajs = true_trajs + model_trajs
    labels = ([f"True r={param['r']}, delay={param['delay']}, x0={param['x_init']}" for param in test_params] 
    + [f"Model r={param['r']}, delay={param['delay']}, x0={param['x_init']}" for param in test_params])
    shifts = [param["delay"] for param in test_params]
    shifts *= 2

    time_series = plot_time_series(trajs, labels, False)
    time_series.savefig(os.path.join(work_dir, "time_series.png"))
    phase_space = plot_phase_space(trajs, labels, shifts, False)
    phase_space.savefig(os.path.join(work_dir, "phase_space.png"))