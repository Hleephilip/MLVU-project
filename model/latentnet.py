import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple

import torch
from choices import *
from config_base import BaseConfig
from torch import nn
from torch.nn import init

from .blocks import *
from .nn import timestep_embedding
from .unet import *


class LatentNetType(Enum):
    none = 'none'
    # injecting inputs into the hidden layers
    skip = 'skip'


class LatentNetReturn(NamedTuple):
    pred: torch.Tensor = None,
    mu: torch.Tensor = None,
    log_sigma: torch.Tensor = None

class ConditioningAugmention(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(ConditioningAugmention, self).__init__()
        self.input_dim = input_dim
        self.intermediate_dim = 512
        self.emb_dim = emb_dim
        self.layer = nn.Sequential(
            nn.Linear(self.input_dim, self.intermediate_dim),
            nn.ReLU(),
            nn.Linear(self.intermediate_dim, self.emb_dim * 2),
        )
    
    def forward(self, x):
        _ = self.layer(x)
        mu, log_sigma = _[:, :self.emb_dim], _[:, self.emb_dim:]

        B = mu.shape[0]
        eps = torch.randn(B, self.emb_dim).to('cuda')
        condition = eps * log_sigma.exp() + mu

        return condition, mu, log_sigma
    
@dataclass
class MLPSkipNetConfig(BaseConfig):
    """
    default MLP for the latent DPM in the paper!
    """
    num_channels: int
    skip_layers: Tuple[int]
    num_hid_channels: int
    num_layers: int
    num_time_emb_channels: int = 64
    activation: Activation = Activation.silu
    use_norm: bool = True
    condition_bias: float = 1
    dropout: float = 0
    last_act: Activation = Activation.none
    num_time_layers: int = 2
    time_last_act: bool = False
    condition_vec_dim: int = None
    x_vec_dim: int = None

    def make_model(self):
        return MLPSkipNet(self)


class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers

    default MLP for the latent DPM in the paper!
    """
    def __init__(self, conf: MLPSkipNetConfig):
        super().__init__()
        self.conf = conf

        layers = []
        for i in range(conf.num_time_layers):
            if i == 0:
                a = conf.num_time_emb_channels
                b = conf.num_channels
            else:
                a = conf.num_channels
                b = conf.num_channels
            layers.append(nn.Linear(a, b))
            if i < conf.num_time_layers - 1 or conf.time_last_act:
                layers.append(conf.activation.get_act())
        self.time_embed = nn.Sequential(*layers)

        self.layers = nn.ModuleList([])
        for i in range(conf.num_layers):
            if i == 0:
                act = conf.activation
                norm = conf.use_norm
                cond = True
                a, b = conf.num_channels, conf.num_hid_channels
                dropout = conf.dropout
            elif i == conf.num_layers - 1:
                act = Activation.none
                norm = False
                cond = False
                a, b = conf.num_hid_channels, conf.num_channels
                dropout = 0
            else:
                act = conf.activation
                norm = conf.use_norm
                cond = True
                a, b = conf.num_hid_channels, conf.num_hid_channels
                dropout = conf.dropout

            if i in conf.skip_layers:
                a += conf.num_channels

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=conf.num_channels,
                    use_cond=cond,
                    condition_bias=conf.condition_bias,
                    dropout=dropout,
                    condition_vec_chans=conf.condition_vec_dim
                ))
        
        self.last_act = conf.last_act.get_act()
        # self.conditioning_aug = ConditioningAugmention(conf.num_channels, conf.num_channels)

    def forward(self, x, t, c, **kwargs):
        t = timestep_embedding(t, self.conf.num_time_emb_channels)
        cond_t = self.time_embed(t)
        # if c is not None: 
        #     c, mu, log_sigma = self.conditioning_aug(c)
        # else:
        #     mu, log_sigma = None, None
        mu, log_sigma = None, None
        h = x
        for i in range(len(self.layers)):
            if i in self.conf.skip_layers:
                # injecting input into the hidden layers
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond_t=cond_t, c=c)
        h = self.last_act(h)
        return LatentNetReturn(h, mu, log_sigma)


class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        use_cond: bool,
        activation: Activation,
        cond_channels: int,
        condition_bias: float = 0,
        dropout: float = 0,
        condition_vec_chans: int = 0
    ):
        super().__init__()
        self.activation = activation
        self.condition_bias = condition_bias
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = activation.get_act()
        if self.use_cond:
            self.linear_emb_t = nn.Linear(cond_channels, out_channels)
            self.cond_t_layers = nn.Sequential(self.act, self.linear_emb_t)

            # self.linear_emb_c = nn.Linear(cond_channels, out_channels)
            self.cond_c_layers_1 = nn.Sequential(self.act, 
                                               nn.Linear(condition_vec_chans, 512),
                                               self.act
                                               )
            self.cond_c_layers_2 = nn.Sequential(nn.Linear(512, 512),
                                               self.act,
                                               nn.Linear(512, 512),
                                               self.act,
                                               nn.Linear(512, 512),
                                                )
            self.cond_c_layers_3 = nn.Sequential(self.act,
                                               nn.Linear(512, out_channels)
                                               )

        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == Activation.relu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                elif self.activation == Activation.lrelu:
                    init.kaiming_normal_(module.weight,
                                         a=0.2,
                                         nonlinearity='leaky_relu')
                elif self.activation == Activation.silu:
                    init.kaiming_normal_(module.weight,
                                         a=0,
                                         nonlinearity='relu')
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond_t, c):
        x = self.linear(x)
        if self.use_cond:
            # print(cond_t.shape) # [B, 768]
            # (n, c) or (n, c * 2)
            cond_t = self.cond_t_layers(cond_t)
            if c is not None:
                c = self.cond_c_layers_1(c)
                _c = self.cond_c_layers_2(c) + c
                c = self.cond_c_layers_3(_c)
                # print(cond_t.shape) # [B, 2048]
                # print(c.shape) # [B, 2048]
            cond = (cond_t, c)

            # scale shift first
            x = x * (self.condition_bias + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x