import json
import os
import matplotlib.pyplot as plt

import torch

from mingpt.model import GPT

def plot_att(att_scores, tokens=None, batch=0, figsize_scale=2.5):
    """
    Plot attention as a grid:
    - rows = layers
    - columns = heads

    att: torch.Tensor of shape (B, n_layer, n_head, T, T)
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

def load_gpt_from_dir(work_dir, map_location="cpu"):
    """
    Input: work_dir (string) containing config.json and model.pt
    Output: GPT model with those params, located at "map_location" and in eval mode
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

def plot_att_from_model(model, input):
    model.eval()
    idx, att_scores = model.generate(input.unsqueeze(0), 1, output_att_scores = True)
    print(idx.squeeze(0))
    plot_att(att_scores, tokens=input)

if __name__ == '__main__':
    # adder_model = load_gpt_from_dir("out/adder")
    # plot_att_from_model(adder_model, torch.tensor([2,4,6,8,2]))

    char_model = load_gpt_from_dir("out/chargpt")
    plot_att_from_model(char_model, torch.tensor([char_model.stoi[s] for s in "O God, O God! "]))
