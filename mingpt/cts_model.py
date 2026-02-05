import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN
from mingpt.model import BlockOutput, Block

# -----------------------------------------------------------------------------

class ContinuousGPT(nn.Module):
    """
    GPT Float Model
    Tensor -> BlockOutput, att_scores.shape = (B, num_layers, nh, T, T)
    """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.input_dim = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C
            
    def __init__(self, config):
        super().__init__()
        assert config.input_dim is not None
        assert config.block_size is not None
        self.input_dim = config.input_dim
        self.block_size = config.block_size

        # set n_layer, n_head, n_embd
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type]) # inputs the nested dict of 3 params to merge_from_dict
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(self.input_dim, config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional embedding
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # Head outputs Mean and Log-Variance (for stability)
        self.regr_head = nn.Linear(config.n_embd, self.input_dim * 2) # use bias so can have nonzero predictions when input 0

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        with torch.no_grad():
            self.regr_head.weight.normal_(std=0.001) # since predicting log variance, sensitive to large weights
            self.regr_head.bias[:self.input_dim].fill_(0) # !!!!! initialize where you expect mean path to be

        # report number of params, including regr_head
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    # allow loading from pretrained later

    # same as before, embedding and regression_head linear layers now have regularization like all other linear layers
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_parameters is recursive we will see the same tensors p many many times
                # but doing it this way (having m that the p belongs to) is needed
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # x, targets of shape (b, t <= block_size, input_dim)
    def forward(self, x, targets=None, output_att_scores=False): # returns mu, sigma for next prediction
        device = x.device
        b, t, x_input_dim = x.size()
        assert x_input_dim == self.input_dim, f"Cannot forward sequence of input dimension {x_input_dim}, model is for dimension {self.input_dim}"
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        # forward the GPT model itself
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # same shape (1, t) and data type (long) as for discrete values, this is expected input into nn.Embedding to account for discrete time index
        tok_emb = self.transformer.wte(x) # (b, t <= block_size, input_dim) -> (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # (1, t) -> (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        att_scores_mult = [] if output_att_scores else None
        for block in self.transformer.h:
            output = block(x, output_att_scores)
            x = output.y
            if output_att_scores:
                att_scores_mult.append(output.att_scores)
        if output_att_scores:
            att_scores_mult = torch.stack(att_scores_mult, dim=1)            
        x = self.transformer.ln_f(x)
        dist_params = self.regr_head(x)
        mu, log_var = torch.chunk(dist_params, 2, dim=-1) # each of shape (b, t, input_dim)
        
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # Gaussian Negative Log Likelihood Loss
            # Using log_var is more numerically stable than raw variance
            # no ignore option, need to make mask if want to ignore anything
            loss = F.gaussian_nll_loss(mu, targets, torch.exp(log_var))

        return BlockOutput(dist_params, att_scores_mult, loss)
    
    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0, output_att_scores=False):
        """
        Take a conditioning sequence of values x (FloatTensor of shape (b,t,input_dim)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Returns attention scores for final token only
        """
        att_scores = None
        for i in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            x_cond = x if x.size(1) <= self.block_size else x[:, -self.block_size:, :]
            # forward the model to get the params for the index in the sequence
            if output_att_scores and i == max_new_tokens - 1:
                output = self(x_cond, output_att_scores = True)
                att_scores = output.att_scores
            else:
                output = self(x_cond)
            mu, log_var = torch.chunk(output.y[:, -1, :], 2, dim=-1)
            # sqrt(var) to get stdev, scale by desired temperature, and sample from the distribution
            std = torch.exp(0.5 * log_var) * temperature
            x_next = torch.normal(mu, std).unsqueeze(1) # (b, input_dim) -> (b, 1, input_dim)
            # append sampled index to the running sequence and continue
            x = torch.cat((x, x_next), dim=1)

        return BlockOutput(x, att_scores)