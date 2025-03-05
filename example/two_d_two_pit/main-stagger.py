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
project_root = current_dir.parent.parent  # 向上两级到 project_working_dir
sys.path.append(str(project_root))  # 将根目录加入模块搜索路径

from pf_pinn import *
from example.two_d_two_pit.configs import *

# from jax import config
# config.update("jax_disable_jit", True)


class PINN(nn.Module):

    def __init__(
        self,
        num_layers,
        hidden_dim,
        out_dim,
        act_name,
        fourier_emb=True,
        arch_name="mlp",
    ):
        super().__init__()

        self.loss_fn_panel = [
            self.loss_pde,
            self.loss_ic,
            self.loss_bc,
            self.loss_irr,
            self.loss_flux,
        ]
        self.pde_name = "ac"
        self.aux_vars = {}
        self.causal_weightor = CausalWeightor(
            num_chunks=CHUNKS, t_range=(0.0, 1.0), pde_name="ac"
        )
        arch = {"mlp": MLP, "modified_mlp": ModifiedMLP}
        self.model = arch[arch_name](
            act_name=act_name,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            fourier_emb=fourier_emb,
        )

    @partial(jit, static_argnums=(0,))
    def ref_sol_bc(self, x, t):
        # x: (x1, x2)
        r = jnp.sqrt((jnp.abs(x[:, 0]) - 0.15) ** 2 + x[:, 1] ** 2)
        # phi = jnp.where(r < 0.05**2, 0, 1)
        # c = jnp.where(r < 0.05**2, 0, 1)
        phi = (r > 0.05).astype(jnp.float32)
        c = phi.copy()
        sol = jnp.stack([phi, c], axis=1)
        return jax.lax.stop_gradient(sol)

    @partial(jit, static_argnums=(0,))
    def ref_sol_ic(self, x, t):
        r = jnp.sqrt((jnp.abs(x[:, 0]) - 0.15) ** 2 + x[:, 1] ** 2)
        phi = (
            1
            - (
                1
                - jnp.tanh(
                    jnp.sqrt(OMEGA_PHI) / jnp.sqrt(2 * ALPHA_PHI) * (r - 0.05) * Lc
                )
            )
            / 2
        )
        h_phi = -2 * phi**3 + 3 * phi**2
        c = h_phi * CSE + (1 - h_phi) * 0.0
        sol = jnp.stack([phi, c], axis=1)
        return jax.lax.stop_gradient(sol)

    def grad(self, func: Callable, argnums: int):
        return jax.grad(
            lambda *args, **kwargs: func(*args, **kwargs).sum(), argnums=argnums
        )

    @partial(jit, static_argnums=(0,))
    def net_u(self, params, x, t):

        def hard_cons(params, x, t):
            phi, cl = nn.tanh(self.model.apply(params, x, t)) / 2 + 0.5
            cl = cl * (1 - CSE + CLE)
            c = (CSE - CLE) * (-2 * phi**3 + 3 * phi**2) + cl
            return jnp.stack([phi, c], axis=0)

        return (
            hard_cons(params, x, t) + hard_cons(params, x * jnp.array([-1, 1]), t)
        ) / 2
        # return hard_cons(params, x, t)

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
        g_phi = phi**2 * (1 - phi) ** 2
        dg_dphi = 4 * phi**3 - 6 * phi**2 + 2 * phi

        jac = jax.jacrev(self.net_u, argnums=(1, 2))
        dphi_dx, dc_dx = jac(params, x, t)[0]
        dphi_dt, dc_dt = jac(params, x, t)[1]

        hess = jax.hessian(self.net_u, argnums=(1))
        d2phi_dx2, d2c_dx2 = hess(params, x, t)

        nabla2phi = d2phi_dx2
        nabla2c = d2c_dx2

        nabla2_hphi = 6 * (phi * (1 - phi) * nabla2phi + (1 - 2 * phi) * dphi_dx**2)

        ch = dc_dt - CH1 * nabla2c + CH1 * (CSE - CLE) * nabla2_hphi
        ac = (
            dphi_dt
            - AC1 * (c - h_phi * (CSE - CLE) - CLE) * (CSE - CLE) * dh_dphi
            + AC2 * dg_dphi
            - AC3 * nabla2phi
        )

        return [ac / AC_PRE_SCALE, ch / CH_PRE_SCALE]

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

        jac_phi_t = jax.jacrev(lambda x, t: self.net_u(params, x, t)[0], argnums=1)
        dphi_dt = jac_phi_t(x, t)

        hess_phi_x = jax.hessian(lambda x, t: self.net_u(params, x, t)[0], argnums=0)
        nabla2phi = jnp.linalg.trace(hess_phi_x(x, t))

        ac = (
            dphi_dt
            - AC1 * (c - h_phi * (CSE - CLE) - CLE) * (CSE - CLE) * dh_dphi
            + AC2 * dg_dphi
            - AC3 * nabla2phi
        )
        return ac.squeeze() / AC_PRE_SCALE

    @partial(jit, static_argnums=(0,))
    def net_ch(self, params, x, t):
        CH1 = 2 * AA * MM * Tc / Lc**2

        # self.net_u : (x, t) --> (phi, c)
        phi, c = self.net_u(params, x, t)

        # jac = jax.jacrev(self.net_u, argnums=(1, 2))
        # dphi_dx, dc_dx = jac(params, x, t)[0]
        # dphi_dt, dc_dt = jac(params, x, t)[1]

        jac_phi_x = jax.jacrev(lambda x, t: self.net_u(params, x, t)[0], argnums=0)
        dphi_dx = jac_phi_x(x, t)

        jac_c_t = jax.jacrev(lambda x, t: self.net_u(params, x, t)[1], argnums=1)
        dc_dt = jac_c_t(x, t)

        # hess = jax.hessian(self.net_u, argnums=(1))
        # hess_phi_x, hess_c_x = hess(params, x, t)
        hess_phi_x = jax.hessian(lambda x, t: self.net_u(params, x, t)[0], argnums=0)(
            x, t
        )
        hess_c_x = jax.hessian(lambda x, t: self.net_u(params, x, t)[1], argnums=0)(
            x, t
        )

        nabla2phi = jnp.linalg.trace(hess_phi_x)
        nabla2c = jnp.linalg.trace(hess_c_x)

        nabla2_hphi = 6 * (
            phi * (1 - phi) * nabla2phi + (1 - 2 * phi) * jnp.sum(dphi_dx**2)
        )

        ch = dc_dt - CH1 * nabla2c + CH1 * (CSE - CLE) * nabla2_hphi

        return ch.squeeze() / CH_PRE_SCALE

    @partial(jit, static_argnums=(0,))
    def net_speed(self, params, x, t):
        jac_dt = jax.jacrev(self.net_u, argnums=2)
        dphi_dt, dc_dt = jac_dt(params, x, t)
        return dphi_dt, dc_dt

    @partial(jit, static_argnums=(0,))
    def net_nabla(self, params, x, t, on="y"):
        idx = 1 if on == "y" else 0
        nabla_phi_part = jax.jacrev(
            lambda x, t: self.net_u(params, x, t)[0], argnums=0
        )(x, t)[idx]
        nabla_c_part = jax.jacrev(lambda x, t: self.net_u(params, x, t)[1], argnums=0)(
            x, t
        )[idx]
        return nabla_phi_part, nabla_c_part

    @partial(jit, static_argnums=(0,))
    def loss_ac(self, params, batch):
        x, t = batch
        # ac, _ = vmap(self.net_pde, in_axes=(None, 0, 0))(params, x, t)
        ac = vmap(self.net_ac, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(ac**2)

    @partial(jit, static_argnums=(0,))
    def loss_ch(self, params, batch):
        x, t = batch
        # _, ch = vmap(self.net_pde, in_axes=(None, 0, 0))(params, x, t)
        ch = vmap(self.net_ch, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(ch**2)

    @partial(jit, static_argnums=(0,))
    def loss_pde(self, params, batch, eps):
        x, t = batch
        pde_name = self.pde_name
        pde_fn = self.net_ac if pde_name == "ac" else self.net_ch
        res = vmap(pde_fn, in_axes=(None, 0, 0))(params, x, t)
        if not CAUSAL_WEIGHT:
            return jnp.mean(res**2)
        else:
            return self.causal_weightor.compute_causal_loss(res, t, eps)

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
    def loss_irr(self, params, batch):
        x, t = batch
        dphi_dt, dc_dt = vmap(self.net_speed, in_axes=(None, 0, 0))(params, x, t)
        # dphi_dt must be negative, use relu to ensure it
        return jnp.mean(jax.nn.relu(dphi_dt)) + jnp.mean(jax.nn.relu(dc_dt))

    @partial(jit, static_argnums=(0,))
    def loss_flux(self, params, batch):
        x, t = batch
        dphi_dy, dc_dy = vmap(self.net_nabla, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(dphi_dy**2) + jnp.mean(dc_dy**2)

    @partial(jit, static_argnums=(0,))
    def compute_losses_and_grads(self, params, batch, eps):
        if len(batch) != len(self.loss_fn_panel):
            raise ValueError(
                "The number of loss functions "
                "should be equal to the number of items in the batch"
            )
        losses = []
        grads = []
        aux_vars = {}
        for idx, (loss_item_fn, batch_item) in enumerate(
            zip(self.loss_fn_panel, batch)
        ):
            if idx == 0:
                (loss_item, aux), grad_item = jax.value_and_grad(
                    loss_item_fn, has_aux=True
                )(params, batch_item, eps)
                aux_vars.update(aux)
            else:
                loss_item, grad_item = jax.value_and_grad(loss_item_fn)(
                    params, batch_item
                )
            losses.append(loss_item)
            grads.append(grad_item)

        return jnp.array(losses), grads, aux_vars

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params, batch, eps):
        losses, grads, aux_vars = self.compute_losses_and_grads(params, batch, eps)

        weights = self.grad_norm_weights(grads)
        weights = jax.lax.stop_gradient(weights)

        return jnp.sum(weights * losses), (losses, weights, aux_vars)

    @partial(jit, static_argnums=(0,))
    def grad_norm_weights(self, grads: list, eps=1e-6):
        grads_flat = [ravel_pytree(grad)[0] for grad in grads]
        grad_norms = [jnp.linalg.norm(grad) for grad in grads_flat]
        grad_norms = jnp.array(grad_norms)
        # grad clipping within [1e-8, 1e8]
        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        weights = jnp.mean(grad_norms) / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        return weights


class Sampler:

    def __init__(
        self,
        n_samples,
        domain=((-0.5, 0.5), (0, 0.5), (0, 1)),
        key=random.PRNGKey(0),
        adaptive_kw={
            "ratio": 10,
            "model": None,
            "state": None,
        },
    ):
        self.n_samples = n_samples
        self.domain = domain
        self.adaptive_kw = adaptive_kw
        self.key = key
        self.mins = [d[0] for d in domain]
        self.maxs = [d[1] for d in domain]
        

    def adaptive_sampling(self, residual_fn):
        adaptive_base = lhs_sampling(
            self.mins, self.maxs, self.n_samples**3 * self.adaptive_kw["ratio"]
        )
        residuals = residual_fn(adaptive_base)
        max_residuals, indices = jax.lax.top_k(
            jnp.abs(residuals), self.n_samples**3 // 5
        )
        return adaptive_base[indices]

    def sample_pde(self):
        # data = lhs_sampling(self.mins, self.maxs, self.n_samples**3)
        data = shfted_grid(
            self.mins,
            self.maxs,
            [self.n_samples, self.n_samples, self.n_samples * 3],
            self.key,
        )
        return data[:, :-1], data[:, -1:]

    def sample_ac(self):
        batch = lhs_sampling(self.mins, self.maxs, self.n_samples**3)

        def residual_fn(batch):
            model = self.adaptive_kw["model"]
            params = self.adaptive_kw["params"]
            x, t = batch[:, :-1], batch[:, -1:]
            return vmap(model.net_ac, in_axes=(None, 0, 0))(params, x, t)

        adaptive_sampling = self.adaptive_sampling(residual_fn)
        data = jnp.concatenate([batch, adaptive_sampling], axis=0)
        return data[:, :-1], data[:, -1:]

    def sample_ch(self):
        batch = lhs_sampling(self.mins, self.maxs, self.n_samples**2)

        def residual_fn(batch):
            model = self.adaptive_kw["model"]
            params = self.adaptive_kw["params"]
            x, t = batch[:, :-1], batch[:, -1:]
            return vmap(model.net_ch, in_axes=(None, 0, 0))(params, x, t)

        adaptive_sampling = self.adaptive_sampling(residual_fn)
        data = jnp.concatenate([batch, adaptive_sampling], axis=0)
        return data[:, :-1], data[:, -1:]

    def sample_ic(self):
        x = lhs_sampling(
            mins=[self.domain[0][0], self.domain[1][0]],
            maxs=[self.domain[0][1], self.domain[1][1]],
            num=self.n_samples**2 * 2,
        )
        x_local = lhs_sampling(
            mins=[-0.3, 0], maxs=[0.3, 0.15], num=self.n_samples**2 * 2
        )
        x = jnp.concatenate([x, x_local], axis=0)
        t = jnp.zeros_like(x[:, 0:1])
        return x, t

    def sample_bc(self):
        self.key, subkey = random.split(self.key)
        # t = random.uniform(subkey, (self.n_samples,),
        #                    minval=self.domain[1][0] + self.domain[1][1] / 10,
        #                    maxval=self.domain[1][1])
        # x = jnp.array([self.domain[0][0], self.domain[0][1]])
        # top: x1 \in (self.domain[0][0], self.domain[0][1]), x2 = self.domain[1][1], t \in (self.domain[2][0], self.domain[2][1])
        x1t = lhs_sampling(
            mins=[self.domain[0][0], self.domain[2][0]],
            maxs=[self.domain[0][1], self.domain[2][1]],
            num=self.n_samples**2 / 5,
        )
        top = jnp.concatenate(
            [x1t[:, 0:1], jnp.ones_like(x1t[:, 0:1]) * self.domain[1][1], x1t[:, 1:2]],
            axis=1,
        )
        x2t = lhs_sampling(
            mins=[self.domain[1][0], self.domain[2][0]],
            maxs=[self.domain[1][1], self.domain[2][1]],
            num=self.n_samples**2 / 5,
        )
        left = jnp.concatenate(
            [jnp.ones_like(x2t[:, 0:1]) * self.domain[0][0], x2t[:, 0:1], x2t[:, 1:2]],
            axis=1,
        )
        right = jnp.concatenate(
            [jnp.ones_like(x2t[:, 0:1]) * self.domain[0][1], x2t[:, 0:1], x2t[:, 1:2]],
            axis=1,
        )

        # local: x1 \in (self.domain[0][0]/20, self.domain[0][1]/20), x2 = self.domain[1][0], t \in (self.domain[2][0] + self.domain[2][1] / 10, self.domain[2][1])
        x1t = lhs_sampling(
            mins=[self.domain[0][0] / 20, self.domain[2][0] + self.domain[2][1] / 10],
            maxs=[self.domain[0][1] / 20, self.domain[2][1]],
            num=self.n_samples**2 / 5,
        )
        local = jnp.concatenate(
            [x1t[:, 0:1], jnp.ones_like(x1t[:, 0:1]) * self.domain[1][0], x1t[:, 1:2]],
            axis=1,
        )
        local_left = local.copy()
        local_right = local.copy()
        local_left = local_left.at[:, 0:1].set(local_left[:, 0:1] - 0.15)
        local_right = local_right.at[:, 0:1].set(local_right[:, 0:1] + 0.15)
        data = jnp.concatenate([top, left, right, local_left, local_right], axis=0)
        return data[:, :-1], data[:, -1:]

    def sample_flux(self):
        x1t = lhs_sampling(
            mins=[self.domain[0][0], self.domain[2][0]],
            maxs=[self.domain[0][1], self.domain[2][1]],
            num=self.n_samples**2 / 2,
        )
        top = jnp.concatenate(
            [x1t[:, 0:1], jnp.ones_like(x1t[:, 0:1]) * self.domain[1][0], x1t[:, 1:2]],
            axis=1,
        )
        return top[:, :-1], top[:, -1:]

    def sample(self, pde_name="ac"):
        if pde_name == "ac":
            data_pde = self.sample_ac()
        elif pde_name == "ch":
            data_pde = self.sample_ch()
        else:
            raise ValueError("Invalid PDE name")
        # data_pde = self.sample_pde()
        return (
            data_pde,
            self.sample_ic(),
            self.sample_bc(),
            data_pde,
            self.sample_flux(),
        )


def create_train_state(model, rng, lr, **kwargs):
    decay = kwargs.get("decay", 0.9)
    decay_every = kwargs.get("decay_every", 1000)
    params = model.init(rng, jnp.ones((1, 2)), jnp.ones((1, 1)))
    scheduler = optax.exponential_decay(lr, decay_every, decay, staircase=True)
    optimizer = optax.adam(scheduler)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@jit
def train_step(state, batch, eps):
    params = state.params
    (weighted_loss, (loss_components, weight_components, aux_vars)), grads = (
        jax.value_and_grad(pinn.loss_fn, has_aux=True, argnums=0)(params, batch, eps)
    )
    new_state = state.apply_gradients(grads=grads)
    return new_state, (weighted_loss, loss_components, weight_components, aux_vars)


init_key = random.PRNGKey(0)
model_key, sampler_key = random.split(init_key)
pinn = PINN(
    num_layers=NUM_LAYERS,
    hidden_dim=HIDDEN_DIM,
    out_dim=OUT_DIM,
    act_name=ACT_NAME,
    fourier_emb=FOURIER_EMB,
    arch_name=ARCH_NAME,
)


state = create_train_state(
    pinn.model, model_key, LR, decay=DECAY, decay_every=DECAY_EVERY
)

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_path = f"{LOG_DIR}/{PREFIX}/{now}"
metrics_tracker = MetricsTracker(log_path)
sampler = Sampler(
    N_SAMPLES,
    domain=DOMAIN,
    key=sampler_key,
    adaptive_kw={"ratio": 2, "model": pinn, "params": state.params},
)
stagger = StaggerSwitch()

start_time = time.time()
for epoch in range(EPOCHS):
    pde_name = stagger.switch(epoch, STAGGER_PERIOD)
    pinn.pde_name = pde_name
    pinn.causal_weightor.pde_name = pde_name

    if epoch % STAGGER_PERIOD == 0:
        sampler.adaptive_kw["params"].update(state.params)
        batch = sampler.sample(pde_name=pde_name)
        
    state, (weighted_loss, loss_components, weight_components, aux_vars) = train_step(
        state, batch, CAUSAL_CONFIGS[pde_name + "_eps"]
    )
    update_causal_eps(aux_vars["causal_weights"], CAUSAL_CONFIGS, pde_name)
    
    if epoch % STAGGER_PERIOD == 0:
        fig, error = evaluate2D(
            pinn,
            state.params,
            jnp.load(f"{DATA_PATH}/mesh_points.npy"),
            DATA_PATH,
            ts=TS,
            Lc=Lc,
            Tc=Tc,
        )

        print(
            f"Epoch: {epoch}, "
            f"Error: {error:.2e}, "
            f"Loss_{pde_name}: {loss_components[0]:.2e}, "
        )
        
        metrics_tracker.register_scalars(
            epoch,
            names=[
                "loss/weighted",
                f"loss/{pde_name}",
                "loss/ic",
                "loss/bc",
                "loss/irr",
                "loss/flux",
                f"weight/{pde_name}",
                "weight/ic",
                "weight/bc",
                "weight/irr",
                "weight/flux",
                "error/error",
            ],
            values=[weighted_loss, *loss_components, *weight_components, error],
        )
        metrics_tracker.register_figure(epoch, fig, "error")
        plt.close(fig)

        fig = pinn.causal_weightor.plot_causal_info(
            pde_name,
            aux_vars["causal_weights"],
            aux_vars["loss_chunks"],
            CAUSAL_CONFIGS[pde_name + "_eps"],
        )
        metrics_tracker.register_figure(epoch, fig, "causal_info")
        plt.close(fig)

        metrics_tracker.flush()


# save the model
params = state.params
model_path = f"{log_path}/model.npz"
params = jax.device_get(params)
jnp.savez(model_path, **params)

end_time = time.time()
print(f"Training time: {end_time - start_time}")
