import time
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import scipy.io as sio
from flax import linen as nn
from flax.training import train_state
from jax import jit, random, vmap
from jax.flatten_util import ravel_pytree

from configs.conf_ac import *
from pf_pinn import *


class PINN(nn.Module):

    def __init__(self, num_layers, hidden_dim, out_dim, act_name,
                 fourier_emb=True, arch_name="mlp"):
        super().__init__()

        # bc: u(-1, t) = u(1, t) = -1
        self.ref_sol_bc = lambda x, t: jnp.full_like(x.shape[0], -1)
        self.ref_sol_ic = lambda x, t: x**2 * jnp.cos(jnp.pi * x)
        self.loss_item_fns = [self.loss_pde, self.loss_ic, self.loss_bc]
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
        # u_t = 0.001 * u_xx + 5 * (u - u^3)
        jac = jax.jacrev(self.net_u, argnums=(2))
        u_t = jac(params, x, t)[0]
        u_xx = jax.hessian(self.net_u, argnums=1)(params, x, t)
        u = self.net_u(params, x, t)
        return u_t - 0.001 * u_xx - 5 * (u - u ** 3)

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
    def compute_losses_and_grads(self, params, batch):
        if len(batch) != len(self.loss_item_fns):
            raise ValueError("The number of loss functions "
                             "should be equal to the number of items in the batch")
        losses = []
        grads = []
        for loss_item_fn, batch_item in zip(self.loss_item_fns, batch):
            loss_item, grad_item = jax.value_and_grad(
                loss_item_fn)(params, batch_item)
            losses.append(loss_item)
            grads.append(grad_item)

        return jnp.array(losses), grads

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params, batch, wights=None):
        losses, grads = self.compute_losses_and_grads(params, batch)

        weights = self.grad_norm_weights(grads)
        weights = jax.lax.stop_gradient(weights)

        return jnp.sum(weights * losses), losses

    @partial(jit, static_argnums=(0,))
    def grad_norm_weights(self, grads: list, eps=1e-8):
        grads_flat = [ravel_pytree(grad)[0] for grad in grads]
        grad_norms = [jnp.linalg.norm(grad) for grad in grads_flat]
        grad_norms = jnp.array(grad_norms)
        return jnp.sum(grad_norms) / (grad_norms + eps)


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
        return mesh_flat(x, t)
        # x, t = jnp.meshgrid(x, t)
        # return x.reshape(-1, 1), t.reshape(-1, 1)

    def sample_ic(self):
        self.key, subkey = random.split(self.key)
        x = random.uniform(subkey, (self.n_samples,),
                           minval=self.domain[0][0],
                           maxval=self.domain[0][1])
        t = jnp.array([self.domain[1][0],])
        return mesh_flat(x, t)

        # x, t = jnp.meshgrid(x, t)
        # return x.reshape(-1, 1), t.reshape(-1, 1)

    def sample_bc(self):
        self.key, subkey = random.split(self.key)
        t = random.uniform(subkey, (self.n_samples,),
                           minval=self.domain[1][0],
                           maxval=self.domain[1][1])
        x = jnp.array([self.domain[0][0], self.domain[0][1]])
        return mesh_flat(x, t)
        # x, t = jnp.meshgrid(x, t)
        # return x.reshape(-1, 1), t.reshape(-1, 1)

    def sample(self):
        return self.sample_pde(), self.sample_ic(), self.sample_bc()


def create_train_state(model, rng, lr, **kwargs):
    decay = kwargs.get("decay", 0.9)
    decay_every = kwargs.get("decay_every", 1000)
    params = model.init(rng, jnp.ones((1, 1)), jnp.ones((1, 1)))
    scheduler = optax.exponential_decay(lr, decay_every, decay, staircase=True)
    optimizer = optax.adam(scheduler)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@jit
def train_step(state, batch):
    params = state.params
    (weighted_loss, loss_components), grads = jax.value_and_grad(
        pinn.loss_fn, has_aux=True, argnums=0)(params, batch)
    new_state = state.apply_gradients(grads=grads)
    return new_state, (weighted_loss, loss_components)


init_key = random.PRNGKey(0)
model_key, sampler_key = random.split(init_key)
sampler = Sampler(N_SAMPLES, key=sampler_key, domain=DOMAIN)
pinn = PINN(
    num_layers=NUM_LAYERS,
    hidden_dim=HIDDEN_DIM,
    out_dim=OUT_DIM,
    act_name=ACT_NAME,
    fourier_emb=FOURIER_EMB,
    arch_name=ARCH_NAME
)


state = create_train_state(pinn.model, model_key, LR,
                           decay=DECAY, decay_every=DECAY_EVERY)

metrics_tracker = MetricsTracker(LOG_DIR, PREFIX)

data = sio.loadmat(DATA_PATH)
x_valid = data["x"].reshape(-1,)
t_valid = data["t"].reshape(-1,)
u_valid = data["u"]
x_valid, t_valid = jnp.meshgrid(x_valid, t_valid)
batch_valid = (x_valid, t_valid)


start_time = time.time()
for epoch in range(EPOCHS):
    if epoch % PAUSE_EVERY == 0:
        batch = sampler.sample()
    state, (weighted_loss, loss_components) = train_step(state, batch)
    if epoch % PAUSE_EVERY == 0:
        fig, error = evaluate1D(pinn, state.params,
                                batch_valid, u_valid,
                                xlim=(0, 1), ylim=(-1, 1),
                                val_range=(-1, 1))

        print(
            f"Epoch: {epoch}, Error: {error}, Loss: {weighted_loss}")
        metrics_tracker.register_scalars(epoch, {
            "loss/weighted": jnp.sum(weighted_loss),
            "loss/pde": loss_components[0],
            "loss/ic": loss_components[1],
            "loss/bc": loss_components[2],
            "error/error": error
        })
        metrics_tracker.register_figure(epoch, fig)
        metrics_tracker.writer.flush()
        plt.close(fig)

end_time = time.time()
print(f"Training time: {end_time - start_time}")
