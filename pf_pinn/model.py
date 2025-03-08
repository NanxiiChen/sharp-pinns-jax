import sys
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, vmap
from jax.flatten_util import ravel_pytree


from pf_pinn import CausalWeightor, MLP, ModifiedMLP


class PINN(nn.Module):

    def __init__(
        self,
        config: object = None,
    ):
        super().__init__()

        self.cfg = config

        self.loss_fn_panel = [
            self.loss_pde,
            self.loss_ic,
            self.loss_bc,
            self.loss_irr,
            # self.loss_flux,
        ]
        self.pde_name = "ac"
        self.aux_vars = {}
        self.causal_weightor = CausalWeightor(
            num_chunks=self.cfg.CAUSAL_CONFIGS["chunks"],
            t_range=(0.0, 1.0),
        )
        arch = {"mlp": MLP, "modified_mlp": ModifiedMLP}
        self.model = arch[self.cfg.ARCH_NAME](
            act_name=self.cfg.ACT_NAME,
            num_layers=self.cfg.NUM_LAYERS,
            hidden_dim=self.cfg.HIDDEN_DIM,
            out_dim=self.cfg.OUT_DIM,
            fourier_emb=self.cfg.FOURIER_EMB,
            emb_scale=self.cfg.EMB_SCALE,
            emb_dim=self.cfg.EMB_DIM
        )

    @partial(jit, static_argnums=(0,))
    def ref_sol_bc(self, x, t):
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def ref_sol_ic(self, x, t):
        raise NotImplementedError

    def grad(self, func: Callable, argnums: int):
        return jax.grad(
            lambda *args, **kwargs: func(*args, **kwargs).sum(), argnums=argnums
        )

    @partial(jit, static_argnums=(0,))
    def net_u(self, params, x, t):
        def hard_cons(params, x, t):
            phi, cl = nn.tanh(self.model.apply(params, x, t)) / 2 + 0.5
            cl = cl * (1 - self.cfg.CSE + self.cfg.CLE)
            c = (self.cfg.CSE - self.cfg.CLE) * (-2 * phi**3 + 3 * phi**2) + cl
            return jnp.stack([phi, c], axis=0)

        return (
            hard_cons(params, x, t) + hard_cons(params, x * jnp.array([-1, 1]), t)
        ) / 2
        # return hard_cons(params, x, t)

    @partial(jit, static_argnums=(0,))
    def net_ac(self, params, x, t):
        AC1 = 2 * self.cfg.AA * self.cfg.LP * self.cfg.Tc
        AC2 = self.cfg.LP * self.cfg.OMEGA_PHI * self.cfg.Tc
        AC3 = self.cfg.LP * self.cfg.ALPHA_PHI * self.cfg.Tc / self.cfg.Lc**2

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
            - AC1
            * (c - h_phi * (self.cfg.CSE - self.cfg.CLE) - self.cfg.CLE)
            * (self.cfg.CSE - self.cfg.CLE)
            * dh_dphi
            + AC2 * dg_dphi
            - AC3 * nabla2phi
        )
        return ac.squeeze() / self.cfg.AC_PRE_SCALE

    @partial(jit, static_argnums=(0,))
    def net_ch(self, params, x, t):
        CH1 = 2 * self.cfg.AA * self.cfg.MM * self.cfg.Tc / self.cfg.Lc**2

        # self.net_u : (x, t) --> (phi, c)
        phi, c = self.net_u(params, x, t)

        jac_phi_x = jax.jacrev(lambda x, t: self.net_u(params, x, t)[0], argnums=0)
        dphi_dx = jac_phi_x(x, t)

        jac_c_t = jax.jacrev(lambda x, t: self.net_u(params, x, t)[1], argnums=1)
        dc_dt = jac_c_t(x, t)

        hess_phi_x, hess_c_x = jax.hessian(self.net_u, argnums=(1))(params, x, t)

        nabla2phi = jnp.linalg.trace(hess_phi_x)
        nabla2c = jnp.linalg.trace(hess_c_x)

        nabla2_hphi = 6 * (
            phi * (1 - phi) * nabla2phi + (1 - 2 * phi) * jnp.sum(dphi_dx**2)
        )

        ch = dc_dt - CH1 * nabla2c + CH1 * (self.cfg.CSE - self.cfg.CLE) * nabla2_hphi

        return ch.squeeze() / self.cfg.CH_PRE_SCALE

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
        if not self.cfg.CAUSAL_WEIGHT:
            return jnp.mean(res**2), {}
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
