import datetime
import sys
import time
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import linen as nn
from flax.training import train_state
from jax import jit, random, vmap
from jax.flatten_util import ravel_pytree

current_dir = Path(__file__).resolve().parent  # 当前文件所在目录 (example1)
project_root = current_dir.parent.parent       # 向上两级到 project_working_dir
sys.path.append(str(project_root))             # 将根目录加入模块搜索路径

from pf_pinn import *
from examples.one_d_active.configs import *



class PINN(nn.Module):

    def __init__(self, num_layers, hidden_dim, out_dim, act_name,
                 fourier_emb=True, arch_name="mlp"):
        super().__init__()

        self.loss_item_fns = [self.loss_ac, self.loss_ch,
                              self.loss_ic, self.loss_bc]
        arch = {"mlp": MLP, "modified_mlp": ModifiedMLP}
        self.model = arch[arch_name](act_name=act_name, num_layers=num_layers,
                                     hidden_dim=hidden_dim, out_dim=out_dim, fourier_emb=fourier_emb)

    @partial(jit, static_argnums=(0,))
    def ref_sol_bc(self, x, t):
        # u(x<0, t) = [1, 1], u(x>0, t) = [0, 0]
        phi = jnp.where(x < 0, 1, 0)
        c = jnp.where(x < 0, 1, 0)
        return jnp.stack([phi, c], axis=0)

    @partial(jit, static_argnums=(0,))
    def ref_sol_ic(self, x, t):
        phi = (1 - jnp.tanh(jnp.sqrt(OMEGA_PHI) /
                            jnp.sqrt(2 * ALPHA_PHI) * x * Lc)) / 2
        h_phi = -2 * phi**3 + 3 * phi**2
        c = h_phi * CSE + (1 - h_phi) * 0.0
        return jnp.stack([phi, c], axis=0)

    def grad(self, func: Callable, argnums: int):
        return jax.grad(lambda *args, **kwargs: func(*args, **kwargs).sum(), argnums=argnums)

    @partial(jit, static_argnums=(0,))
    def net_u(self, params, x, t):
        # return nn.tanh(self.model.apply(params, x, t)) / 2 + 0.5
        sol = self.model.apply(params, x, t)
        phi, cl = nn.tanh(sol) / 2 + 0.5
        cl = cl * (1 - CSE + CLE)
        c = (CSE - CLE) * (-2*phi**3 + 3*phi**2) + cl
        return jnp.stack([phi, c], axis=0)

    @partial(jit, static_argnums=(0,))
    def net_pde(self, params, x, t):
        AC1 = 2 * AA * LP * Tc
        AC2 = LP * OMEGA_PHI * Tc
        AC3 = LP * ALPHA_PHI * Tc / Lc**2
        CH1 = 2 * AA * MM * Tc / Lc**2

        # self.net_u : (x, t) --> (phi, c)
        phi, c = self.net_u(params, x, t)
        h_phi = -2 * phi**3 + 3 * phi**2
        dh_dphi = -6 * phi**2 + 6 * phi
        g_phi = phi**2 * (1 - phi)**2
        dg_dphi = 4 * phi**3 - 6 * phi**2 + 2 * phi

        jac = jax.jacrev(self.net_u, argnums=(1, 2))
        dphi_dx, dc_dx = jac(params, x, t)[0]
        dphi_dt, dc_dt = jac(params, x, t)[1]

        hess = jax.hessian(self.net_u, argnums=(1))
        d2phi_dx2, d2c_dx2 = hess(params, x, t)

        nabla2phi = d2phi_dx2
        nabla2c = d2c_dx2

        nabla2_hphi = 6 * (
            phi * (1 - phi) * nabla2phi
            + (1 - 2*phi) * dphi_dx**2
        )

        ch = dc_dt - CH1 * nabla2c + CH1 * (CSE - CLE) * nabla2_hphi
        ac = dphi_dt - AC1 * (c - h_phi*(CSE-CLE) - CLE) * (CSE-CLE) * dh_dphi \
            + AC2 * dg_dphi - AC3 * nabla2phi

        return [ac/AC_PRE_SCALE, ch/CH_PRE_SCALE]

    @partial(jit, static_argnums=(0,))
    def net_ac(self, params, x, t):
        AC1 = 2 * AA * LP * Tc
        AC2 = LP * OMEGA_PHI * Tc
        AC3 = LP * ALPHA_PHI * Tc / Lc**2

        # self.net_u : (x, t) --> (phi, c)
        phi, c = self.net_u(params, x, t)
        h_phi = -2 * phi**3 + 3 * phi**2
        dh_dphi = -6 * phi**2 + 6 * phi
        dg_dphi = 4 * phi**3 - 6 * phi**2 + 2 * phi

        jac = jax.jacrev(lambda x, t: self.net_u(params, x, t)[0],
                         argnums=1)
        dphi_dt = jac(x, t)

        hess = jax.hessian(lambda x, t: self.net_u(params, x, t)[0],
                           argnums=0)
        d2phi_dx2 = hess(x, t)
        nabla2phi = d2phi_dx2

        ac = dphi_dt - AC1 * (c - h_phi*(CSE-CLE) - CLE) * (CSE-CLE) * dh_dphi \
            + AC2 * dg_dphi - AC3 * nabla2phi
        return ac.squeeze() / AC_PRE_SCALE

    @partial(jit, static_argnums=(0,))
    def net_ch(self, params, x, t):
        CH1 = 2 * AA * MM * Tc / Lc**2

        # self.net_u : (x, t) --> (phi, c)
        phi, c = self.net_u(params, x, t)

        jac = jax.jacrev(self.net_u, argnums=(1, 2))
        dphi_dx, dc_dx = jac(params, x, t)[0]
        dphi_dt, dc_dt = jac(params, x, t)[1]

        hess = jax.hessian(self.net_u, argnums=(1))
        d2phi_dx2, d2c_dx2 = hess(params, x, t)

        nabla2phi = d2phi_dx2
        nabla2c = d2c_dx2

        nabla2_hphi = 6 * (
            phi * (1 - phi) * nabla2phi
            + (1 - 2*phi) * dphi_dx**2
        )

        ch = dc_dt - CH1 * nabla2c + CH1 * (CSE - CLE) * nabla2_hphi

        return ch.squeeze() / CH_PRE_SCALE

    @partial(jit, static_argnums=(0,))
    def loss_ac(self, params, batch):
        x, t = batch
        # ac, _ = vmap(self.net_pde, in_axes=(None, 0, 0))(params, x, t)
        ac = vmap(self.net_ac, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(ac ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_ch(self, params, batch):
        x, t = batch
        # _, ch = vmap(self.net_pde, in_axes=(None, 0, 0))(params, x, t)
        ch = vmap(self.net_ch, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(ch ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_ic(self, params, batch):
        x, t = batch
        u = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean((u - self.ref_sol_ic(x, t)) ** 2)

    @partial(jit, static_argnums=(0,))
    def loss_bc(self, params, batch):
        x, t = batch
        u = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean((u - self.ref_sol_bc(x, t)) ** 2)

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
    def loss_fn(self, params, batch,):
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


# class Sampler:

#     def __init__(self, n_samples,
#                  domain=((-0.5, 0.5), (0, 1)),
#                  key=random.PRNGKey(0),
#                  adaptive_base_ratio=10):
#         self.n_samples = n_samples
#         self.domain = domain
#         self.key = key
#         self.adaptive_ratio = adaptive_base_ratio

#     def sample_pde(self):
#         self.key, subkey = random.split(self.key)
#         x = random.uniform(subkey, (self.n_samples,),
#                            minval=self.domain[0][0],
#                            maxval=self.domain[0][1])
#         t = random.uniform(subkey, (self.n_samples,),
#                            minval=self.domain[1][0],
#                            maxval=self.domain[1][1])
#         return mesh_flat(x, t)

#     def sample_ic(self):
#         self.key, subkey = random.split(self.key)
#         x = random.uniform(subkey, (self.n_samples,),
#                            minval=self.domain[0][0],
#                            maxval=self.domain[0][1])
#         x_local = x / 10
#         x = jnp.concatenate([x, x_local], axis=0)
#         t = jnp.array([self.domain[1][0],])
#         return mesh_flat(x, t)

#     def sample_bc(self):
#         self.key, subkey = random.split(self.key)
#         t = random.uniform(subkey, (self.n_samples,),
#                            minval=self.domain[1][0],
#                            maxval=self.domain[1][1])
#         x = jnp.array([self.domain[0][0], self.domain[0][1]])
#         return mesh_flat(x, t)

#     def sample(self):
#         return self.sample_pde(), self.sample_pde(), self.sample_ic(), self.sample_bc()


class Sampler:

    def __init__(self, n_samples,
                 domain=((-0.5, 0.5), (0, 1)),
                 key=random.PRNGKey(0),
                 adaptive_kw={
                     "ratio": 10,
                     "model": None,
                     "state": None,
                 }):
        self.n_samples = n_samples
        self.domain = domain
        self.adaptive_kw = adaptive_kw
        self.key = key
        self.mins = [d[0] for d in domain]
        self.maxs = [d[1] for d in domain]

    def adaptive_sampling(self, residual_fn):
        adaptive_base = lhs_sampling(self.mins, self.maxs,
                                     self.n_samples**2 * self.adaptive_kw["ratio"])
        residuals = residual_fn(adaptive_base)
        max_residuals, indices = jax.lax.top_k(jnp.abs(residuals),
                                               self.n_samples**2)
        return adaptive_base[indices]

    def sample_pde(self):
        data = lhs_sampling(self.mins, self.maxs, self.n_samples**2)
        return data[:, :-1], data[:, -1:]

    def sample_ac(self):
        batch = lhs_sampling(self.mins, self.maxs, self.n_samples**2)

        def loss_fn(batch):
            model = self.adaptive_kw["model"]
            params = self.adaptive_kw["state"].params
            x, t = batch[:, :-1], batch[:, -1:]
            return vmap(model.net_ac, in_axes=(None, 0, 0))(params, x, t)

        adaptive_sampling = self.adaptive_sampling(loss_fn)
        data = jnp.concatenate([batch, adaptive_sampling], axis=0)
        return data[:, :-1], data[:, -1:]

    def sample_ch(self):
        batch = lhs_sampling(self.mins, self.maxs, self.n_samples**2)

        def residual_fn(batch):
            model = self.adaptive_kw["model"]
            params = self.adaptive_kw["state"].params
            x, t = batch[:, :-1], batch[:, -1:]
            return vmap(model.net_ch, in_axes=(None, 0, 0))(params, x, t)

        adaptive_sampling = self.adaptive_sampling(residual_fn)
        data = jnp.concatenate([batch, adaptive_sampling], axis=0)
        return data[:, :-1], data[:, -1:]

    def sample_ic(self):
        self.key, subkey = random.split(self.key)
        x = random.uniform(subkey, (self.n_samples,),
                           minval=self.domain[0][0],
                           maxval=self.domain[0][1])
        x_local = x / 10
        x = jnp.concatenate([x, x_local], axis=0)
        t = jnp.array([self.domain[1][0],])
        return mesh_flat(x, t)

    def sample_bc(self):
        self.key, subkey = random.split(self.key)
        t = random.uniform(subkey, (self.n_samples,),
                           minval=self.domain[1][0],
                           maxval=self.domain[1][1])
        x = jnp.array([self.domain[0][0], self.domain[0][1]])
        return mesh_flat(x, t)

    def sample(self):
        return self.sample_ac(), self.sample_ch(), self.sample_ic(), self.sample_bc()
        # return self.sample_pde(), self.sample_pde(), self.sample_ic(), self.sample_bc()


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
sampler = Sampler(
    N_SAMPLES,
    domain=DOMAIN,
    key=sampler_key,
    adaptive_kw={
        "ratio": 10,
        "model": pinn,
        "state": state
    }
)

data = jnp.load(DATA_PATH)
x_valid = data["x"].reshape(-1,) / Lc
t_valid = data["t"].reshape(-1,) / Tc
phi_valid = data["phi"]
x_valid, t_valid = jnp.meshgrid(x_valid, t_valid)
batch_valid = (x_valid, t_valid)


start_time = time.time()
for epoch in range(EPOCHS):
    if epoch % PAUSE_EVERY == 0:
        batch = sampler.sample()
    state, (weighted_loss, loss_components) = train_step(state, batch)
    if epoch % PAUSE_EVERY == 0:
        fig, error = evaluate1D(pinn, state.params,
                                batch_valid, phi_valid,
                                xlim=(-0.5, 0.5), ylim=(0, 1),
                                val_range=(0, 1))

        print(
            f"Epoch: {epoch}, Error: {error}, Loss: {weighted_loss}")
        metrics_tracker.register_scalars(epoch, {
            "loss/weighted": jnp.sum(weighted_loss),
            "loss/ac": loss_components[0],
            "loss/ch": loss_components[1],
            "loss/ic": loss_components[2],
            "loss/bc": loss_components[3],
            "error/error": error
        })
        metrics_tracker.register_figure(epoch, fig)
        metrics_tracker.writer.flush()
        plt.close(fig)

end_time = time.time()
print(f"Training time: {end_time - start_time}")
