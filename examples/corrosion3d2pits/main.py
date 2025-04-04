"""
Sharp-PINNs for pitting corrosion with 2d-1pit
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
from examples.corrosion3d2pits.configs import Config as cfg


# from jax import config
# config.update("jax_disable_jit", True)


class PFSampler(Sampler):

    def __init__(
        self,
        n_samples,
        domain=((-0.4, 0.4), (-0.4, 0.4), (0, 0.4), (0, 1)),
        key=random.PRNGKey(0),
        adaptive_kw={
            "ratio": 10,
            "num": 5000,
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


    def sample_pde(self):
        key, self.key = random.split(self.key)
        data = shifted_grid(
            self.mins,
            self.maxs,
            [
                int(self.n_samples * 1.4),
                self.n_samples,
                self.n_samples // 2,
                self.n_samples * 2,
            ],
            key,
        )
        return data[:, :-1], data[:, -1:]

    def sample_ic(self):
        key, self.key = random.split(self.key)
        x = lhs_sampling(
            mins=[self.domain[0][0], self.domain[1][0], self.domain[2][0]],
            maxs=[self.domain[0][1], self.domain[1][1], self.domain[2][1]],
            num=4000,
            key=key,
        )
        x_local = lhs_sampling(
            mins=[-0.2, -0.2, 0],
            maxs=[
                0.2,
                0.2,
                0.2,
            ],
            num=4000,
            key=self.key,
        )
        x = jnp.concatenate([x, x_local], axis=0)
        t = jnp.zeros_like(x[:, 0:1])
        return x, t

    def sample_bc(self):
        key, self.key = random.split(self.key)

        local = lhs_sampling(
            mins=[-0.05, -0.05, 0, self.domain[3][0]],
            maxs=[0.05, 0.05, 0.05, self.domain[3][1]],
            num=self.n_samples**2 * 5,
            key=key,
        )
        local_left = local.copy()
        local_right = local.copy()

        local_left = local_left.at[:, 0:1].set(local[:, 0:1] - 0.2)
        local_right = local_right.at[:, 0:1].set(local[:, 0:1] + 0.2)

        yzts = lhs_sampling(
            mins=[self.domain[1][0], self.domain[2][0], self.domain[3][0]],
            maxs=[self.domain[1][1], self.domain[2][1], self.domain[3][1]],
            num=self.n_samples * 5,
            key=key,
        )
        xmin_yzts = jnp.concatenate(
            [
                jnp.full((yzts.shape[0], 1), self.domain[0][0]),
                yzts,
            ],
            axis=1,
        )
        xmax_yzts = jnp.concatenate(
            [
                jnp.full((yzts.shape[0], 1), self.domain[0][1]),
                yzts,
            ],
            axis=1,
        )
        xzts = lhs_sampling(
            mins=[self.domain[0][0], self.domain[2][0], self.domain[3][0]],
            maxs=[self.domain[0][1], self.domain[2][1], self.domain[3][1]],
            num=self.n_samples * 5,
            key=key,
        )
        ymin_xzts = jnp.concatenate(
            [
                xzts[:, 0:1],
                jnp.full((xzts.shape[0], 1), self.domain[1][0]),
                xzts[:, 1:],
            ],
            axis=1,
        )
        ymax_xzts = jnp.concatenate(
            [
                xzts[:, 0:1],
                jnp.full((xzts.shape[0], 1), self.domain[1][1]),
                xzts[:, 1:],
            ],
            axis=1,
        )
        xyts = lhs_sampling(
            mins=[self.domain[0][0], self.domain[1][0], self.domain[3][0]],
            maxs=[self.domain[0][1], self.domain[1][1], self.domain[3][1]],
            num=self.n_samples * 5,
            key=key,
        )
        # zmin_xyts = jnp.concatenate([
        #     xyts[:, 0:2],
        #     jnp.full((xyts.shape[0], 1), self.domain[2][0]),
        #     xyts[:, 2:],
        # ], axis=1)
        zmax_xyts = jnp.concatenate(
            [
                xyts[:, 0:2],
                jnp.full((xyts.shape[0], 1), self.domain[2][1]),
                xyts[:, 2:],
            ],
            axis=1,
        )

        data = jnp.concatenate(
            [
                xmin_yzts,
                xmax_yzts,
                ymin_xzts,
                ymax_xzts,
                zmax_xyts,
                local_left,
                local_right,
            ],
            axis=0,
        )
        return data[:, :-1], data[:, -1:]
    
    def sample_flux(self):
        key, self.key = random.split(self.key)
        xyts = lhs_sampling(
            mins=[self.domain[0][0], self.domain[1][0], self.domain[3][0]],
            maxs=[self.domain[0][1], self.domain[1][1], self.domain[3][1]],
            num=self.n_samples * 5,
            key=key,
        )
        data = jnp.concatenate(
            [
                xyts[:, 0:2],
                jnp.full((xyts.shape[0], 1), self.domain[2][0]),
                xyts[:, 2:],
            ],
            axis=1,
        )
        return data[:, :-1], data[:, -1:]

    def sample(self, pde_name="ac"):
        return (
            self.sample_pde_rar(),
            self.sample_ic(),
            self.sample_bc(),
            #self.sample_flux(),
            self.sample_pde(),
        )


class PFPINN(PINN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn_panel = [
            self.loss_pde,
            self.loss_ic,
            self.loss_bc,
            #self.loss_flux,
            self.loss_irr,
        ]
        self.flux_idx = 2

    @partial(jit, static_argnums=(0,))
    def ref_sol_bc(self, x, t):
        # x: (x1, x2)
        r = r = jnp.sqrt((jnp.abs(x[0]) - 0.20) ** 2 + x[1] ** 2 + x[2] ** 2)
        phi = jnp.where(r < 0.10, 0, 1)
        c = jnp.where(r < 0.10, 0, 1)
        sol = jnp.stack([phi, c], axis=-1)
        return jax.lax.stop_gradient(sol)

    @partial(jit, static_argnums=(0,))
    def ref_sol_ic(self, x, t):
        r = jnp.sqrt((jnp.abs(x[0]) - 0.20) ** 2 + x[1] ** 2 + x[2] ** 2)
        phi = (
            1
            - (
                1
                - jnp.tanh(
                    jnp.sqrt(cfg.OMEGA_PHI)
                    / jnp.sqrt(2 * cfg.ALPHA_PHI)
                    * (r - 0.10)
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
        "model": pinn,
        "params": state.params,
        "num": cfg.ADAPTIVE_SAMPLES,
    },
)
stagger = StaggerSwitch(pde_names=["ac", "ch"], stagger_period=cfg.STAGGER_PERIOD)

start_time = time.time()
for epoch in range(cfg.EPOCHS):
    pde_name = stagger.decide_pde()
    loss_fn = pinn.loss_fn_ac if pde_name == "ac" else pinn.loss_fn_ch
    
    if epoch % cfg.STAGGER_PERIOD == 0:
        sampler.adaptive_kw["params"].update(state.params)
        batch = sampler.sample(pde_name=pde_name)
        # print(f"Epoch: {epoch}, PDE: {pde_name}")

    state, (weighted_loss, loss_components, weight_components, aux_vars) = train_step(
        loss_fn,
        state,
        batch,
        cfg.CAUSAL_CONFIGS[pde_name + "_eps"],
    )
    if cfg.CAUSAL_WEIGHT:
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

        fig, error = evaluate3D(
            pinn,
            state.params,
            jnp.load(f"{cfg.DATA_PATH}/mesh_points.npy"),
            cfg.DATA_PATH,
            ts=cfg.TS,
            Lc=cfg.Lc,
            Tc=cfg.Tc,
            xlim=cfg.DOMAIN[0],
            ylim=cfg.DOMAIN[1],
            zlim=cfg.DOMAIN[2],
            box_aspect=(4, 2, 1),
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
                #"loss/flux",
                "loss/irr",
                f"weight/{pde_name}",
                "weight/ic",
                "weight/bc",
                #"weight/flux",
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
