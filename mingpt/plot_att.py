import json
import os
import matplotlib.pyplot as plt
import numpy as np

import torch

from mingpt.model import GPT
from mingpt.cts_model import ContinuousGPT

def plot_att(att_scores, tokens=None, figsize_scale=2.5):
    """
    Plot attention scores in grid, row = layer, col = head
    att_scores: torch.Tensor of shape (n_layer, n_head, T, T) (note model outputs size (B, n_layer, n_head, T, T))
    tokens: List[String labels] of len T
    """

    att_scores = att_scores.cpu()  # (n_layer, n_head, T, T)
    n_layer, n_head, T, _ = att_scores.shape
    if tokens is not None:
        assert T == len(tokens), f"Number of labels {len(tokens)} does not match size of attention scores {T}"

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
                ax.set_xticks([]) # remove ticks
                ax.set_yticks([])

            if l == 0:
                ax.set_title(f"Head {h}", fontsize=10)
            if h == 0:
                ax.set_ylabel(f"Layer {l}", fontsize=10)

    # shared colorbar
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02).set_label("Attention Weight")
    plt.show()

def load_gpt_from_dir(work_dir, cts_model=False, map_location="cpu"):
    """
    String (path to dir containing config.json and model.pt of model config and model params) -> 
    GPT or ContinuousGPT class model located at "map_location" and in eval mode
    Prints 
    WARNING: ignores the args.txt file taking in command line inputs, didn't want to spend time figuring out how to parse that file
    """
    
    config_path = os.path.join(work_dir, "config.json")
    model_path = os.path.join(work_dir, "model.pt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json in {work_dir}")
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = ContinuousGPT.get_default_config() if cts_model else GPT.get_default_config()
    config.merge_from_dict(config_dict["model"])
    model = ContinuousGPT(config) if cts_model else GPT(config)
    print(config)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=map_location)
        model.load_state_dict(state_dict)
    else:
        print("Warning: model.pt not found, returning randomly initialized model")
    model.eval()
    return model

def plot_att_from_input(model, input, tokens=None):
    """
    Takes in model and input tensor (of the shape expected by forward fn w/o batch, on the same device)
    Returns tensor containing input with next prediction
    Plots attention scores
    """
    if tokens is None: # formats for list of floats, doesn't handle list of vectors
        tokens = [f"{x:.3f}" if isinstance(x, float) else str(x) for x in input.tolist()]
    model.eval()
    output = model.generate(input.unsqueeze(0), 1, output_att_scores = True) # add batch dim to input to model
    print(tokens)
    plot_att(output.att_scores[0], tokens=tokens) # remove batch dim from output
    return output.y[0]



if __name__ == '__main__':
    # adder_model = load_gpt_from_dir("out/adder")
    # for input in [
    #     [2,4,6,8,2]
    # ]:
    #     adder_input = torch.tensor(input)
    #     output = plot_att_from_input(adder_model, adder_input)
    #     print(output.tolist())


    # char_model = load_gpt_from_dir("out/chargpt")
    # with open("out/chargpt/stoi.json") as f:
    #     stoi = json.load(f)
    #     itos = {i: ch for ch, i in stoi.items()}
    # for input in [
    #     "O God, O God! "
    # ]:
    #     char_input = torch.tensor([stoi[s] for s in input])
    #     output = plot_att_from_input(char_model, char_input, input)
    #     print([itos[int(i)] for i in output.tolist()])


    delay_model = load_gpt_from_dir("out/delaygpt", cts_model=True)
    for input in [
        [0.01435208, 0.01408787, 0.03138163, 0.06992334, 0.15306761, 0.32174402,
                0.61583967, 0.94399509], #9057, train example
        [0.98396119, 0.46809989, 0.01696755, 0.0203966,  0.04531416, 0.10032119,
               0.216452,  0.44010625] #19872, test example
    ]:
        delay_input = torch.tensor(input).unsqueeze(1) # shape (t, input_dim)
        delay_tokens = [f"{x:.3f}" for x in input]
        output = plot_att_from_input(delay_model, delay_input, delay_tokens)
        print(f"{x:.3f}" for x in output.squeeze(1).tolist())


