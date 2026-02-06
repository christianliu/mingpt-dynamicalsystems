import os

import torch
import numpy as np

from mingpt.plot_att import load_gpt_from_dir, plot_att_from_input
from helpers.delaydiff import delayed_logistic_mult, plot_time_series, plot_phase_space

if __name__ == '__main__':

    # get trained model
    work_dir = "saved params/run-2" # "out/delaygpt"
    delay_model_1 = load_gpt_from_dir(work_dir, cts_model=True)
    block_size = delay_model_1.block_size

    # get true trajectories
    compare_n = int(1e4)
    test_params = [ # used first 7 rows for model_1, last 7 rows for model_2
        # {"r": 2.26, "delay": 1, "x_init": [0.1, 0.2]},
        # {"r": 2.26, "delay": 1, "x_init": [0.1, 0.175]},
        # {"r": 2.26, "delay": 1, "x_init": [0.1, 0.1]},
        # {"r": 2.26, "delay": 1, "x_init": [0.1, 0.15]},
        {"r": 2.26, "delay": 1, "x_init": [0.2, 0.2]},
        {"r": 2.26, "delay": 1, "x_init": [0.3, 0.3]},
        {"r": 2.26, "delay": 1, "x_init": [0.5, 0.5]},
        {"r": 1.46, "delay": 3, "x_init": [0.05, 0.05, 0.05, 0.05]},
        {"r": 1.46, "delay": 3, "x_init": [0.1, 0.2, 0.1, 0.2]},
        {"r": 1.46, "delay": 3, "x_init": [0.1, 0.15, 0.1, 0.15]},
        {"r": 1.46, "delay": 3, "x_init": [0.2, 0.2, 0.2, 0.2]}
    ]
    true_trajs = delayed_logistic_mult(test_params, compare_n) # fn: List[Dict{delay:, r:, x_init:}] -> List[np.array of shape (n,)]
    
    ############### get model traj on Colab, comment out when not on Colab ###################
    # # List[np.array of shape (block_size,)] -> tensor of shape (num_traj, block_size, 1)
    # model_input = torch.from_numpy(np.stack([x[:block_size] for x in true_trajs]).astype(np.float32)).unsqueeze(2) # must be type float, print(type(true_trajs[0][0]))
    # model_output = delay_model_1.generate(model_input, compare_n - block_size).y
    # # tensor of shape (num_traj, block_size, 1) -> List[List of len n] -> List[np.array of shape (n,)]
    # model_trajs = [np.array(x) for x in model_output.squeeze(2).tolist()]
    # np.savez(os.path.join(work_dir, "model_output.npz"), model_trajs)
    ##########################################################################################

    with np.load(os.path.join(work_dir, "model_output.npz")) as file_data:
        model_trajs = file_data['arr_0']
    model_trajs = [model_trajs[i] for i in range(model_trajs.shape[0])]

    # plot
    trajs = true_trajs + model_trajs
    labels = ([f"True r={param['r']}, delay={param['delay']}, x0={param['x_init']}" for param in test_params] 
    + [f"Model r={param['r']}, delay={param['delay']}, x0={param['x_init']}" for param in test_params])
    shifts = [param["delay"] for param in test_params]
    shifts *= 2

    time_series = plot_time_series(trajs[5:7] + trajs[12:], labels[5:7] + labels[12:], False)
    phase_space = plot_phase_space(trajs[5:7] + trajs[12:], labels[5:7] + labels[12:], shifts[5:7] + shifts[12:], False)

    for input in [
        true_trajs[1][0:80].astype(np.float32),
        true_trajs[4][0:80].astype(np.float32) 
    ]:
        delay_input = torch.tensor(input).unsqueeze(1) # shape (t, input_dim)
        delay_tokens = [f"{x:.3f}" for x in input]
        output = plot_att_from_input(delay_model_1, delay_input, delay_tokens, max_layers=1)
        print(f"{x:.3f}" for x in output.squeeze(1).tolist())