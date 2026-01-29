import json
import os
import matplotlib.pyplot as plt
import numpy as np

import torch

from mingpt.model import GPT

def plot_att(att_scores, tokens=None, batch=0, figsize_scale=2.5):
    """
    Plot attention scores in grid, one row per layer and one col per head
    att_scores: torch.Tensor of shape (B, n_layer, n_head, T, T)
    tokens: optional list of token strings (length T)
    """

    att_scores = att_scores[batch].cpu()  # (n_layer, n_head, T, T)
    n_layer, n_head, T, _ = att_scores.shape

    fig, axes = plt.subplots(
        n_layer,
        n_head,
        figsize=(figsize_scale * n_head, figsize_scale * n_layer),
        squeeze=False,
        constrained_layout=True
    )

    for l in range(n_layer):
        for h in range(n_head):
            ax = axes[l, h]
            im = ax.imshow(att_scores[l, h], cmap="viridis")

            if tokens is not None:
                ax.set_xticks(range(T))
                ax.set_yticks(range(T))
                ax.set_xticklabels(tokens, rotation=90, fontsize=6)
                ax.set_yticklabels(tokens, fontsize=6)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            if l == 0:
                ax.set_title(f"Head {h}", fontsize=10)
            if h == 0:
                ax.set_ylabel(f"Layer {l}", fontsize=10)

    # shared colorbar
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02).set_label("Attention Weight")

    plt.show()

def load_gpt_from_dir(work_dir, map_location="cpu"): # prints model config and returns a model of that config and params
    """
    work_dir: string containing config.json and model.pt
    output: GPT class model with those params, located at "map_location" and in eval mode
    WARNING: ignores the args.txt file taking in command line inputs, didn't want to spend time figuring out how to parse that file
    """
    
    config_path = os.path.join(work_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in {work_dir}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = GPT.get_default_config()
    config.merge_from_dict(config_dict["model"])
    print(config)
    model = GPT(config)

    model_path = os.path.join(work_dir, "model.pt")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=map_location)
        model.load_state_dict(state_dict)
    else:
        print("Warning: model.pt not found, returning randomly initialized model")
    model.eval()
    return model

def plot_att_from_adder_model(model, input):
    model.eval()
    output = model.generate(torch.tensor(input).unsqueeze(0), 1, output_att_scores = True)
    print(output.y.squeeze(0))
    plot_att(output.att_scores, tokens=input)

def plot_att_from_char_model(model, input, stoi, itos):
    model.eval()
    output = model.generate(torch.tensor([stoi[s] for s in input]).unsqueeze(0), 1, output_att_scores = True)
    print([itos[int(i)] for i in output.y.squeeze(0)])
    plot_att(output.att_scores, tokens=input)

if __name__ == '__main__':
    # adder_model = load_gpt_from_dir("out/adder")
    # plot_att_from_adder_model(adder_model, [2,4,6,8,2])

    # char_model = load_gpt_from_dir("out/chargpt")
    # with open("out/chargpt/stoi.json") as f:
    #     stoi = json.load(f)
    #     itos = {i: ch for ch, i in stoi.items()}
    # plot_att_from_char_model(char_model, "O God, O God! ", stoi, itos)

    delay_model = load_gpt_from_dir("out/delaygpt")
    train_ex = np.array([0.01435208, 0.01408787, 0.03138163, 0.06992334, 0.15306761, 0.32174402,
                0.61583967, 0.94399509]) #9057
    test_ex = np.array([0.98396119, 0.46809989, 0.01696755, 0.0203966,  0.04531416, 0.10032119,
               0.216452,  0.44010625]) #19872
    plot_att_from_adder_model(delay_model, torch.tensor(train_ex*10000, dtype=torch.long))
    plot_att_from_adder_model(delay_model, torch.tensor(test_ex*10000, dtype=torch.long))


# 0.1        0.1        0.2034     0.4137156  0.74481881 0.98688697
#  0.56914707 0.01686693 0.01642379 0.0364917  0.08111676 0.17663408
#  0.36681178 0.68256591 0.9767555  0.7007254  0.03681091 0.02489745
#  0.05419695 0.11943554 0.25529524 0.50805685 0.85507632 0.95066661
#  0.31136948 0.03471566 0.05402816 0.11786474 0.25198259 0.50235896
#  0.84924755 0.95512217 0.32541064 0.03300441 0.0503176  0.10996459
#  0.23601505 0.47473956 0.81968816 0.97304247 0.39651944 0.02415755
#  0.03294767 0.07266292 0.15880758 0.33282604 0.63273388 0.95404446
#  0.79187735 0.082244   0.03868406 0.08023571 0.17431801 0.36234915
#  0.67615848 0.97440585 0.71314993 0.04125058 0.02674197 0.05794381
#  0.12745106 0.2713493  0.53509013 0.88115998 0.92583154 0.24865839
#  0.0416803  0.07077448 0.15328356 0.32190308 0.61598703 0.94399953
#  0.81926822 0.10368746 0.04235154 0.08579009 0.18567424 0.38362423
#  0.7060129  0.9834825  0.65343644 0.02439249 0.01910502 0.04212414
#  0.09338174 0.20215274 0.41420233 0.74686264 0.98877349 0.56566786
#  0.01435208 0.01408787 0.03138163 0.06992334 0.15306761 0.32174402
#  0.61583967 0.94399509 0.81957874 0.10373498
# [0.1        0.2        0.4068     0.7354944  0.98602733 0.58943004
#  0.01861316 0.01727093 0.03830579 0.08507592 0.18490647 0.38233637
#  0.70430617 0.98315494 0.65701103 0.02501231 0.01938842 0.04272184
#  0.09467938 0.20483398 0.41909535 0.75314587 0.98876342 0.55162155
#  0.01400825 0.01419506 0.03163143 0.07047227 0.15422947 0.32399489
#  0.61929724 0.94614471 0.81405178 0.09908065 0.04163795 0.0847781
#  0.18362074 0.37980141 0.70074011 0.98219154 0.66428301 0.02673548
#  0.02028475 0.04461789 0.09879099 0.21330591 0.43444705 0.77241586
#  0.98726307 0.50778904 0.01461695 0.01625984 0.03621012 0.08050424
#  0.17535153 0.36439107 0.67911766 0.97553634 0.70745319 0.03911358
#  0.02586017 0.05615803 0.12363505 0.2637238  0.52232735 0.86914446
#  0.93827638 0.27747977 0.03870715 0.06320474 0.13731367 0.29071464
#  0.56679793 0.90856854 0.88952191 0.18380645 0.04589288 0.08465389
#  0.18253767 0.37761244 0.69762571 0.98127745 0.67057155 0.02837386
#  0.02112458 0.04638693 0.10261988 0.22116283 0.4485357  0.78949999
#  0.98396119 0.46809989 0.01696755 0.0203966  0.04531416 0.10032119
#  0.216452   0.44010625 0.77934828 0.98615604]