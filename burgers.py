import time
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from jax import random, jit, vmap
from jax.nn.initializers import glorot_normal, zeros, normal
import optax
import matplotlib.pyplot as plt
from jax.flatten_util import ravel_pytree
from typing import Callable
from functools import partial
jax.config.update("jax_default_matmul_precision", "high")


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
    emb_dim: int = 64

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel', normal(self.emb_scale),
                            (x.shape[-1], self.emb_dim))
        return jnp.concatenate([jnp.sin(jnp.pi * jnp.dot(x, kernel)),
                                jnp.cos(jnp.pi * jnp.dot(x, kernel))], axis=-1)


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
            t_emb = FourierEmbedding()(t)
            x_emb = FourierEmbedding()(x)
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
            t_emb = FourierEmbedding()(t)
            x_emb = FourierEmbedding()(x)
            x = jnp.concatenate([x_emb, t_emb], axis=-1)
        else:
            x = jnp.concatenate([x, t], axis=-1)

        u = Dense(x.shape[-1], self.hidden_dim)(x)
        v = Dense(x.shape[-1], self.hidden_dim)(x)
        u = self.act_fn(u)
        v = self.act_fn(v)

        for _ in range(self.num_layers):
            x = Dense(x.shape[-1], self.hidden_dim)(x)
            x = self.act_fn(x)
            x = x * u + (1 - x) * v

        return Dense(self.hidden_dim, self.out_dim)(x)


class PINN(nn.Module):

    def __init__(self, num_layers, hidden_dim, out_dim, act_name,
                 fourier_emb=True, arch_name="mlp"):
        super().__init__()

        self.ref_sol_bc = lambda x, t: jnp.zeros(x.shape[0]).reshape(-1, 1)
        self.ref_sol_ic = lambda x, t: -jnp.sin(jnp.pi * x).reshape(-1, 1)
        arch = {"mlp": MLP, "modified_mlp": ModifiedMLP}
        self.model = arch[arch_name](act_name=act_name, num_layers=num_layers,
                                     hidden_dim=hidden_dim, out_dim=out_dim, fourier_emb=fourier_emb)

    def grad(self, func: Callable, argnums: int):
        return jax.grad(lambda *args, **kwargs: func(*args, **kwargs).sum(), argnums=argnums)

    @partial(jit, static_argnums=(0,))
    def net_u(self, params, x, t):
        return nn.tanh(self.model.apply(params, x, t))

    @partial(jit, static_argnums=(0,))
    def net_pde(self, params, x, t):
        # u_t + u * u_x = 0.01 / pi * u_xx, u(-1, t) = u(1, t) = 0, u(x, 0) = -sin(pi * x)
        jac = jax.jacrev(self.net_u, argnums=(1, 2))
        u_x, u_t = jac(params, x, t)
        u_xx = jax.hessian(self.net_u, argnums=1)(params, x, t)
        u = self.net_u(params, x, t)
        return u_t + u_x * u - 0.01 / jnp.pi * u_xx

    @partial(jit, static_argnums=(0,))
    def loss_pde(self, params, batch):
        x, t = batch
        pde = vmap(self.net_pde, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(pde ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_ic(self, params, batch):
        x, t = batch
        u = self.net_u(params, x, t)
        return jnp.mean((u - self.ref_sol_ic(x, t)) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_bc(self, params, batch):
        x, t = batch
        u_bc = self.net_u(params, x, t)
        return jnp.mean((u_bc - self.ref_sol_bc(x, t)) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params, batch,):
        batch_pde, batch_ic, batch_bc = batch
        loss_pde = self.loss_pde(params, batch_pde)
        loss_ic = self.loss_ic(params, batch_ic)
        loss_bc = self.loss_bc(params, batch_bc)
        return loss_pde + loss_ic*10 + loss_bc, [loss_pde, loss_ic, loss_bc]
        # loss_pde, grad_pde = jax.value_and_grad(self.loss_pde)(params, batch_pde)
        # loss_ic, grad_ic = jax.value_and_grad(self.loss_ic)(params, batch_ic)
        # loss_bc, grad_bc = jax.value_and_grad(self.loss_bc)(params, batch_bc)

        # weights = self.grad_norm_weights([grad_pde, grad_ic, grad_bc])
        # losses = jnp.array([loss_pde, loss_ic, loss_bc])
        # return jnp.sum(weights * losses)

    def grad_norm_weights(self, grads: list):
        grads_flat = [ravel_pytree(grad)[0] for grad in grads]
        grad_norms = [jnp.linalg.norm(grad) for grad in grads_flat]
        grad_norms = jnp.array(grad_norms)
        return jnp.sum(grad_norms) / grad_norms


class Sampler:

    def __init__(self, n_samples, domain=((-1, 1), (0, 1)), key=random.PRNGKey(0)):
        self.n_samples = n_samples
        self.domain = domain
        self.key = key

    def sample_pde(self):
        self.key, subkey = random.split(self.key)
        x = random.uniform(subkey, (self.n_samples,),
                           minval=self.domain[0][0],
                           maxval=self.domain[0][1])
        t = random.uniform(subkey, (self.n_samples,),
                           minval=self.domain[1][0],
                           maxval=self.domain[1][1])
        x, t = jnp.meshgrid(x, t)
        return x.reshape(-1, 1), t.reshape(-1, 1)

    def sample_ic(self):
        self.key, subkey = random.split(self.key)
        x = random.uniform(subkey, (self.n_samples,),
                           minval=self.domain[0][0],
                           maxval=self.domain[0][1])
        t = jnp.array([self.domain[1][0],])
        x, t = jnp.meshgrid(x, t)
        return x.reshape(-1, 1), t.reshape(-1, 1)

    def sample_bc(self):
        self.key, subkey = random.split(self.key)
        t = random.uniform(subkey, (self.n_samples,),
                           minval=self.domain[1][0],
                           maxval=self.domain[1][1])
        x = jnp.array([self.domain[0][0], self.domain[0][1]])
        x, t = jnp.meshgrid(x, t)
        return x.reshape(-1, 1), t.reshape(-1, 1)

    def sample(self):
        return self.sample_pde(), self.sample_ic(), self.sample_bc()


epochs = 50000
n_samples = 100
lr = 5e-4
scheduler = optax.exponential_decay(lr, 1000, 0.9, staircase=True)
key = random.PRNGKey(0)
sampler = Sampler(n_samples, key=key)
pinn = PINN(num_layers=4, hidden_dim=40,
            out_dim=1, act_name="tanh",
            fourier_emb=True, arch_name="mlp")
optim = optax.adam(scheduler)

# Initialize model parameters and training state
params = pinn.model.init(key, jnp.ones((1, 1)), jnp.ones((1, 1)))
state = train_state.TrainState.create(
    apply_fn=pinn.model.apply, params=params, tx=optim)


@jit
def train_step(state, batch):
    params = state.params
    (loss_sum, loss_components), grads = jax.value_and_grad(
        pinn.loss_fn, has_aux=True, argnums=0)(params, batch)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss_components


losses = []
acc = []
data = jnp.load("Burgers.npz")
x_valid = data["x"].reshape(-1,)
t_valid = data["t"].reshape(-1,)
u_valid = data["usol"].T
x_valid, t_valid = jnp.meshgrid(x_valid, t_valid)
x_valid = x_valid.reshape(-1, 1)
t_valid = t_valid.reshape(-1, 1)
batch_valid = (x_valid, t_valid)
u_valid = u_valid.reshape(-1, 1)

start_time = time.time()
for epoch in range(epochs):
    if epoch % 100 == 0:
        batch = sampler.sample()
    state, loss_components = train_step(state, batch)
    if epoch % 100 == 0:
        u_pred = pinn.net_u(state.params, x_valid, t_valid)
        error = jnp.mean((u_pred - u_valid) ** 2)
        print(f"Epoch: {epoch}, Error: {error}", end=", ")
        losses.append(loss_components)
        acc.append(error)

        print(f"Training time: {time.time() - start_time}")


fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()
# Plot the loss curve
ax = axes[0]
losses = jnp.array(losses)
ax.plot(losses[:, 0], label="PDE loss")
ax.plot(losses[:, 1], label="IC loss")
ax.plot(losses[:, 2], label="BC loss")
ax.set(xlabel="Epochs", ylabel="Loss", yscale="log")
ax.set_title("Loss curve")
ax.legend()

# Plot the accuracy curve
ax = axes[1]
ax.plot(acc, label="Accuracy u")
ax.set(xlabel="Epochs", ylabel="Accuracy", yscale="log")
ax.set_title("Accuracy curve")
ax.legend()


# plot the solution
u_pred = pinn.net_u(state.params, x_valid, t_valid)
ax = axes[2]
ax.scatter(t_valid, x_valid, c=u_pred, cmap="coolwarm",
           label="Predicted", vmin=-1, vmax=1)
ax.set(xlabel="t", ylabel="x", title="Predicted solution")


ax = axes[3]
ax.scatter(t_valid, x_valid, c=u_valid, cmap="coolwarm",
           label="Reference", vmin=-1, vmax=1)
ax.set(xlabel="t", ylabel="x", title="Reference solution")


plt.savefig("burgers.png", bbox_inches="tight", dpi=300)
plt.show()
