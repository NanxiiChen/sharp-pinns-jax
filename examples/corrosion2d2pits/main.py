"""
Sharp-PINNs for pitting corrosion with 2d-2pits
"""

import datetime
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, random, vmap
import orbax.checkpoint as ocp

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from pf_pinn import *
from examples.corrosion2d2pits.configs import Config as cfg


# from jax import config
# config.update("jax_disable_jit", True)


class PFSampler(Sampler):

    def __init__(
        self,
        n_samples,
        domain=((-0.5, 0.5), (0, 0.5), (0, 1)),
        key=None,
        adaptive_kw=None,
    ):
        self.n_samples = n_samples
        self.domain = domain
        self.key = key if key is not None else random.PRNGKey(0)
        self.adaptive_kw = (
            adaptive_kw
            if adaptive_kw is not None
            else {
                "ratio": 10,
                "num": 5000,
            }
        )
        self.mins = [d[0] for d in domain]
        self.maxs = [d[1] for d in domain]

    def sample_ic(self):
        key, self.key = random.split(self.key)
        x = lhs_sampling(
            mins=[self.domain[0][0], self.domain[1][0]],
            maxs=[self.domain[0][1], self.domain[1][1]],
            num=self.n_samples**2 * 5,
            key=key,
        )
        x_local = lhs_sampling(
            mins=[-0.4, 0], maxs=[0.4, 0.2], num=self.n_samples**2 * 10, key=key
        )
        x = jnp.concatenate([x, x_local], axis=0)
        t = jnp.zeros_like(x[:, 0:1])
        return x, t

    def sample_bc(self):
        key, self.key = random.split(self.key)

        x1t = lhs_sampling(
            mins=[self.domain[0][0], self.domain[2][0]],
            maxs=[self.domain[0][1], self.domain[2][1]],
            num=self.n_samples**2 // 5,
            key=key,
        )
        top = jnp.concatenate(
            [x1t[:, 0:1], jnp.ones_like(x1t[:, 0:1]) * self.domain[1][1], x1t[:, 1:2]],
            axis=1,
        )
        x2t = lhs_sampling(
            mins=[self.domain[1][0], self.domain[2][0]],
            maxs=[self.domain[1][1], self.domain[2][1]],
            num=self.n_samples**2 // 5,
            key=key,
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
            num=self.n_samples**2 // 5,
            key=key,
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
        key, self.key = random.split(self.key)
        x1t = lhs_sampling(
            mins=[self.domain[0][0], self.domain[2][0]],
            maxs=[self.domain[0][1], self.domain[2][1]],
            num=self.n_samples**2 // 2,
            key=key,
        )
        data = jnp.concatenate(
            [x1t[:, 0:1], jnp.ones_like(x1t[:, 0:1]) * self.domain[1][0], x1t[:, 1:2]],
            axis=1,
        )
        return data[:, :-1], data[:, -1:]


class PFPINN(PINN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn_panel = [
            self.loss_pde,
            self.loss_ic,
            self.loss_bc,
            self.loss_flux,
            self.loss_irr,
        ]
        self.flux_idx = 1

    def ref_sol_bc(self, x, t):
        # x: (x1, x2)
        r = jnp.sqrt((jnp.abs(x[0]) - 0.15) ** 2 + x[1] ** 2)
        phi = (r > 0.05).astype(jnp.float32)
        c = phi.copy()
        sol = jnp.stack([phi, c], axis=-1)
        return jax.lax.stop_gradient(sol)

    def ref_sol_ic(self, x, t):
        r = jnp.sqrt((jnp.abs(x[0]) - 0.15) ** 2 + x[1] ** 2)
        phi = (
            1
            - (
                1
                - jnp.tanh(
                    jnp.sqrt(cfg.OMEGA_PHI)
                    / jnp.sqrt(2 * cfg.ALPHA_PHI)
                    * (r - 0.05)
                    * cfg.Lc
                )
            )
            / 2
        )
        h_phi = -2 * phi**3 + 3 * phi**2
        c = h_phi * cfg.CSE + (1 - h_phi) * cfg.CLE
        sol = jnp.stack([phi, c], axis=-1)
        return jax.lax.stop_gradient(sol)


pinn = PFPINN(config=cfg)

init_key = random.PRNGKey(0)
model_key, sampler_key = random.split(init_key)
state = create_train_state(
    pinn.model,
    model_key,
    cfg.LR,
    decay=cfg.DECAY,
    decay_every=cfg.DECAY_EVERY,
    xdim=len(cfg.DOMAIN) - 1,
)
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_path = f"{cfg.LOG_DIR}/{cfg.PREFIX}/{now}"
metrics_tracker = MetricsTracker(log_path)
ckpt = ocp.StandardCheckpointer()
sampler = PFSampler(
    cfg.N_SAMPLES,
    domain=cfg.DOMAIN,
    key=sampler_key,
    adaptive_kw={
        "ratio": cfg.ADAPTIVE_BASE_RATE,
        "num": cfg.ADAPTIVE_SAMPLES,
    },
)
stagger = StaggerSwitch(
    pde_names=["ac", "ch"],
    stagger_period=cfg.STAGGER_PERIOD,
)

start_time = time.time()
for epoch in range(cfg.EPOCHS):
    pde_name = stagger.decide_pde()
    loss_fn = pinn.loss_fn_ac if pde_name == "ac" else pinn.loss_fn_ch

    if epoch % cfg.STAGGER_PERIOD == 0:
        # sampler.adaptive_kw["params"].update(state.params)
        batch = sampler.sample(fns=[pinn.net_ac, pinn.net_ch], params=state.params)

    state, (weighted_loss, loss_components, weight_components, aux_vars) = train_step(
        loss_fn,
        state,
        batch,
        cfg.CAUSAL_CONFIGS[pde_name + "_eps"],
    )

    if cfg.CAUSAL_WEIGHT:
        # update_causal_eps(aux_vars["causal_weights"], cfg.CAUSAL_CONFIGS, pde_name)
        cfg.CAUSAL_CONFIGS.update(
            update_causal_eps(
                aux_vars["causal_weights"],
                cfg.CAUSAL_CONFIGS,
                pde_name,
            )
        )
    stagger.step_epoch()

    if epoch % cfg.STAGGER_PERIOD == 0:

        ckpt.save(log_path + f"/model-{epoch}", state)

        fig, error = evaluate2D(
            pinn,
            state.params,
            jnp.load(f"{cfg.DATA_PATH}/mesh_points.npy"),
            cfg.DATA_PATH,
            ts=cfg.TS,
            Lc=cfg.Lc,
            Tc=cfg.Tc,
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
                "loss/flux",
                "loss/irr",
                f"weight/{pde_name}",
                "weight/ic",
                "weight/bc",
                "weight/flux",
                "weight/irr",
                "error/error",
            ],
            values=[weighted_loss, *loss_components, *weight_components, error],
        )
        metrics_tracker.register_figure(epoch, fig, "error")
        plt.close(fig)

        if cfg.CAUSAL_WEIGHT:
            fig = pinn.causal_weightor.plot_causal_info(
                pde_name,
                aux_vars["causal_weights"],
                aux_vars["loss_chunks"],
                cfg.CAUSAL_CONFIGS[pde_name + "_eps"],
            )
            metrics_tracker.register_figure(epoch, fig, "causal_info")
            plt.close(fig)

        metrics_tracker.flush()


end_time = time.time()
print(f"Training time: {end_time - start_time}")
