
from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import glorot_normal, normal, zeros


class Dense(nn.Module):
    in_features: int
    out_features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros

    def setup(self):
        self.kernel = self.param('kernel', self.kernel_init,
                                 (self.in_features, self.out_features))
        self.bias = self.param('bias', self.bias_init,
                               (self.out_features,))

    @nn.compact
    def __call__(self, x):
        return jnp.dot(x, self.kernel) + self.bias


class FourierEmbedding(nn.Module):
    emb_scale: float = 2.0
    emb_dim: int = 32

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel', normal(self.emb_scale),
                            (x.shape[-1], self.emb_dim))
        return jnp.concatenate([jnp.sin(jnp.pi*jnp.dot(x, kernel)),
                                jnp.cos(jnp.pi*jnp.dot(x, kernel))], axis=-1)
        
# class ExponentialEmbedding(nn.Module):
#     emb_scale: tuple = (1/2, 3/2)
#     emb_dim: int = 32
    
#     @nn.compact
#     def __call__(self, x):
#         low, high = self.emb_scale
#         # the initial kernel is a linspace from low to high
#         kernel = jnp.linspace(low, high, self.emb_dim)


#         return jnp.dot(x, jnp.ones((x.shape[-1], self.emb_dim))) ** kernel
        
    


class MLP(nn.Module):
    act_name: str = "tanh"
    num_layers: int = 4
    hidden_dim: int = 64
    out_dim: int = 2
    fourier_emb: bool = True

    def setup(self):
        self.act_fn = getattr(nn, self.act_name)

    @nn.compact
    def __call__(self, x, t):

        if self.fourier_emb:
            # separate the spatial and temporal coordinates
            t_emb = FourierEmbedding(emb_scale=1.0)(t)
            x_emb = FourierEmbedding(emb_scale=2.0)(x)
            x = jnp.concatenate([x_emb, t_emb], axis=-1)
        else:
            x = jnp.concatenate([x, t], axis=-1)

        x = Dense(x.shape[-1], self.hidden_dim)(x)
        for _ in range(self.num_layers - 1):
            x = Dense(self.hidden_dim, self.hidden_dim)(x)
            x = self.act_fn(x)
        return Dense(self.hidden_dim, self.out_dim)(x)


class ModifiedMLP(nn.Module):
    act_name: str = "tanh"
    num_layers: int = 4
    hidden_dim: int = 64
    out_dim: int = 2
    fourier_emb: bool = True

    def setup(self):
        self.act_fn = getattr(nn, self.act_name)

    @nn.compact
    def __call__(self, x, t):

        if self.fourier_emb:
            # separate the spatial and temporal coordinates
            t_emb = FourierEmbedding(emb_scale=2.0/5)(t)
            x_emb = FourierEmbedding(emb_scale=2.0)(x)
            x = jnp.concatenate([x_emb, t_emb], axis=-1)
        else:
            x = jnp.concatenate([x, t], axis=-1)

        u = Dense(x.shape[-1], self.hidden_dim)(x)
        v = Dense(x.shape[-1], self.hidden_dim)(x)
        u = self.act_fn(u)
        v = self.act_fn(v)

        for _ in range(self.num_layers):
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = nn.tanh(x)
            x = x * u + (1 - x) * v

        return Dense(self.hidden_dim, self.out_dim)(x)
