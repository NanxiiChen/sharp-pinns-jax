from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import linen as nn
from flax.training import train_state
from jax import jit, random, vmap
from jax.nn.initializers import glorot_normal, zeros

# jax.config.update("jax_default_matmul_precision", "high")


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


class MLP(nn.Module):
    act_name: str = "tanh"
    num_layers: int = 4
    in_dim: int = 1
    hidden_dim: int = 64
    out_dim: int = 2

    def setup(self):
        self.act_fn = getattr(nn, self.act_name)

    @nn.compact
    def __call__(self, x):
        x = Dense(self.in_dim, self.hidden_dim)(x)
        for _ in range(self.num_layers - 1):
            x = Dense(self.hidden_dim, self.hidden_dim)(x)
            x = self.act_fn(x)
        x = Dense(self.hidden_dim, self.out_dim)(x)
        return x


class PINN(nn.Module):

    def __init__(self, in_dim, num_layers, hidden_dim, out_dim, act_name):
        super().__init__()

        self.ref_sol = lambda x: jnp.sin(jnp.pi * x)
        self.ref_sol_bc = lambda _: jnp.zeros((2, 1))
        self.model = MLP(act_name=act_name, num_layers=num_layers,
                         in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)

    def grad(self, func: Callable, argnums: int):
        return jax.grad(lambda *args, **kwargs: func(*args, **kwargs).sum(), argnums=argnums)

    def net_u(self, params, x):
        return self.model.apply(params, x)

    def net_pde(self, params, x):
        # - \Delta u = f, s.t. u(-1) = u(1) = 0
        # heessian on `x``
        u_xx = jax.hessian(self.net_u, argnums=1)
        f = jnp.pi ** 2 * jnp.sin(jnp.pi * x)
        return u_xx(params, x) + f

    def loss_fn(self, params, batch):
        x, x_bc = batch
        u_bc = self.net_u(params, x_bc)
        pde = vmap(self.net_pde, in_axes=(None, 0))(params, x)
        loss = jnp.mean((u_bc - self.ref_sol_bc(x)) ** 2) + jnp.mean(pde ** 2)
        return loss

    def accuracy(self, params, batch):
        x, x_bc = batch
        u = self.net_u(params, x)
        u_bc = self.net_u(params, x_bc)
        acc_u = jnp.mean((u - self.ref_sol(x)) ** 2)
        acc_bc = jnp.mean((u_bc - self.ref_sol_bc(x)) ** 2)
        return acc_u, acc_bc


class Sampler:

    def __init__(self, n_samples, domain=(-1, 1), key=random.PRNGKey(0)):
        self.n_samples = n_samples
        self.domain = domain
        self.key = key

    def sample(self):
        self.key, subkey = random.split(self.key)
        x = random.uniform(subkey, (self.n_samples, 1),
                           minval=self.domain[0], maxval=self.domain[1])
        x_bc = jnp.array([self.domain[0], self.domain[1]]).reshape(-1, 1)

        return x, x_bc


epochs = 10000
n_samples = 1000
lr = 5e-4
key = random.PRNGKey(0)
sampler = Sampler(n_samples, key=key)
pinn = PINN(in_dim=1, num_layers=4, hidden_dim=50,
            out_dim=1, act_name="tanh")
optim = optax.adam(lr)

# Initialize model parameters and training state
params = pinn.model.init(key, jnp.ones((1, 1)))
state = train_state.TrainState.create(
    apply_fn=pinn.model.apply, params=params, tx=optim)


@jit
def train_step(state, batch):
    params = state.params
    loss, grads = jax.value_and_grad(pinn.loss_fn)(params, batch)
    acc_u, acc_bc = pinn.accuracy(params, batch)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, acc_u, acc_bc


losses = []
acc_us = []
acc_bcs = []
for epoch in range(epochs):
    batch = sampler.sample()
    state, loss, acc_u, acc_bc = train_step(state, batch)
    losses.append(loss)
    acc_us.append(acc_u)
    acc_bcs.append(acc_bc)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, "
              f"Loss: {loss}, "
              f"Acc_u: {acc_u}, "
              f"Acc_bc: {acc_bc}")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot the loss curve
ax = axes[0]
ax.plot(losses)
ax.set(xlabel="Epochs", ylabel="Loss", yscale="log")
ax.set_title("Loss curve")

# Plot the accuracy curve
ax = axes[1]
ax.plot(acc_us, label="Accuracy u")
ax.plot(acc_bcs, label="Accuracy bc")
ax.set(xlabel="Epochs", ylabel="Accuracy", yscale="log")
ax.set_title("Accuracy curve")
ax.legend()


# plot the solution
x = jnp.linspace(-1, 1, 100).reshape(-1, 1)
u = pinn.net_u(state.params, x)
ax = axes[2]
ax.plot(x, u, label="Predicted")
ax.plot(x, pinn.ref_sol(x), label="Reference")
ax.set(xlabel="x", ylabel="u")
ax.set_title("Solution curve")

plt.show()
