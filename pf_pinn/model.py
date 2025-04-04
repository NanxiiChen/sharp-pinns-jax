from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import jit, vmap


from . import CausalWeightor, MLP, ModifiedMLP


class PINN(nn.Module):

    def __init__(
        self,
        config: object = None,
    ):
        super().__init__()

        self.cfg = config
        self.flux_idx = 1
        self.loss_fn_panel = [
            self.loss_pde,
            self.loss_ic,
            self.loss_bc,
            self.loss_flux,
            self.loss_irr,
        ]
        self.causal_weightor = CausalWeightor(
            num_chunks=self.cfg.CAUSAL_CONFIGS["chunks"],
            t_range=self.cfg.DOMAIN[-1],
        )
        arch = {"mlp": MLP, "modified_mlp": ModifiedMLP}
        self.model = arch[self.cfg.ARCH_NAME](
            act_name=self.cfg.ACT_NAME,
            num_layers=self.cfg.NUM_LAYERS,
            hidden_dim=self.cfg.HIDDEN_DIM,
            out_dim=self.cfg.OUT_DIM,
            fourier_emb=self.cfg.FOURIER_EMB,
            emb_scale=self.cfg.EMB_SCALE,
            emb_dim=self.cfg.EMB_DIM,
        )

        # pre-compile the loss functions
        self.loss_fn_ac = partial(self.loss_fn, pde_name="ac")
        self.loss_fn_ch = partial(self.loss_fn, pde_name="ch")

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

    def hard_cons(self, params, x, t):
        phi, cl = nn.tanh(self.model.apply(params, x, t)) / 2 + 0.5
        cl = cl * (1 - self.cfg.CSE + self.cfg.CLE)
        c = (self.cfg.CSE - self.cfg.CLE) * (-2 * phi**3 + 3 * phi**2) + cl
        return jnp.stack([phi, c], axis=0)

    @partial(jit, static_argnums=(0,))
    def net_u(self, params, x, t):

        # def one_sided(params, x, t):
        #     return nn.tanh(self.model.apply(params, x, t)) / 2 + 0.5

        # return (
        #     one_sided(params, x, t) + one_sided(params, x * jnp.array([-1, 1]), t)
        # ) / 2

        if self.cfg.ASYMMETRIC:
            return (
                self.hard_cons(params, x, t)
                + self.hard_cons(
                    params, x * jnp.array([-1] + [1] * (x.shape[0] - 1)), t
                )
            ) / 2
        else:
            return self.hard_cons(params, x, t)

    def laplacian(self, fn, x, t, argnums=0):
        grad_fn = jax.grad(fn, argnums=argnums)
        return jnp.trace(jax.jacrev(grad_fn, argnums=argnums)(x, t))

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
        dphi_dt = jac_phi_t(x, t)[0]

        hess_phi_x = jax.hessian(lambda x, t: self.net_u(params, x, t)[0], argnums=0)
        lap_phi = jnp.trace(hess_phi_x(x, t))
        # lap_phi = self.laplacian(lambda x, t: self.net_u(params, x, t)[0], x, t)

        ac = (
            dphi_dt
            - AC1
            * (c - h_phi * (self.cfg.CSE - self.cfg.CLE) - self.cfg.CLE)
            * (self.cfg.CSE - self.cfg.CLE)
            * dh_dphi
            + AC2 * dg_dphi
            - AC3 * lap_phi
        )
        return ac / self.cfg.AC_PRE_SCALE

    @partial(jit, static_argnums=(0,))
    def net_ch(self, params, x, t):
        CH1 = 2 * self.cfg.AA * self.cfg.MM * self.cfg.Tc / self.cfg.Lc**2

        # self.net_u : (x, t) --> (phi, c)
        phi, _ = self.net_u(params, x, t)

        jac_phi_x = jax.jacrev(lambda x, t: self.net_u(params, x, t)[0], argnums=0)
        nabla_phi = jac_phi_x(x, t)

        jac_c_t = jax.jacrev(lambda x, t: self.net_u(params, x, t)[1], argnums=1)
        dc_dt = jac_c_t(x, t)[0]

        hess_phi_x = jax.hessian(lambda x, t: self.net_u(params, x, t)[0], argnums=0)(
            x, t
        )
        hess_c_x = jax.hessian(lambda x, t: self.net_u(params, x, t)[1], argnums=0)(
            x, t
        )
        lap_phi = jnp.trace(hess_phi_x)
        lap_c = jnp.trace(hess_c_x)

        # lap_phi = self.laplacian(lambda x, t: self.net_u(params, x, t)[0], x, t)
        # lap_c = self.laplacian(lambda x, t: self.net_u(params, x, t)[1], x, t)

        lap_h_phi = 6 * (
            phi * (1 - phi) * lap_phi + (1 - 2 * phi) * jnp.sum(nabla_phi**2)
        )

        ch = dc_dt - CH1 * lap_c + CH1 * (self.cfg.CSE - self.cfg.CLE) * lap_h_phi

        return ch / self.cfg.CH_PRE_SCALE

    def net_speed(self, params, x, t):
        jac_dt = jax.jacrev(self.net_u, argnums=2)
        dphi_dt, dc_dt = jac_dt(params, x, t)
        return dphi_dt, dc_dt

    def net_nabla(
        self,
        params,
        x,
        t,
    ):
        idx = self.flux_idx
        nabla_phi_part = jax.jacrev(
            lambda x, t: self.net_u(params, x, t)[0], argnums=0
        )(x, t)[idx]
        nabla_c_part = jax.jacrev(lambda x, t: self.net_u(params, x, t)[1], argnums=0)(
            x, t
        )[idx]
        return nabla_phi_part, nabla_c_part

    # def loss_ac(self, params, batch, eps):
    #     x, t = batch
    #     ac = vmap(self.net_ac, in_axes=(None, 0, 0))(params, x, t)
    #     if not self.cfg.CAUSAL_WEIGHT:
    #         return jnp.mean(ac**2), {}
    #     else:
    #         return self.causal_weightor.compute_causal_loss(ac, t, eps)

    # def loss_ch(self, params, batch, eps):
    #     x, t = batch
    #     ch = vmap(self.net_ch, in_axes=(None, 0, 0))(params, x, t)
    #     if not self.cfg.CAUSAL_WEIGHT:
    #         return jnp.mean(ch**2), {}
    #     else:
    #         return self.causal_weightor.compute_causal_loss(ch, t, eps)


    @partial(jit, static_argnums=(0, 4))
    def loss_pde(self, params, batch, eps, pde_name: str):
        x, t = batch
        residual = jax.lax.cond(
            pde_name == "ac",
            lambda operand: vmap(self.net_ac, in_axes=(None, 0, 0))(params, *operand),
            lambda operand: vmap(self.net_ch, in_axes=(None, 0, 0))(params, *operand),
            operand=(x, t),
        )
        if not self.cfg.CAUSAL_WEIGHT:
            return jnp.mean(residual**2), {}
        else:
            return self.causal_weightor.compute_causal_loss(residual, t, eps)

    def loss_ic(self, params, batch):
        x, t = batch
        u = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        ref = vmap(self.ref_sol_ic, in_axes=(0, 0))(x, t)
        return jnp.mean((u - ref) ** 2)

    def loss_bc(self, params, batch):
        x, t = batch
        u = vmap(self.net_u, in_axes=(None, 0, 0))(params, x, t)
        ref = vmap(self.ref_sol_bc, in_axes=(0, 0))(x, t)
        return jnp.mean((u - ref) ** 2)

    def loss_irr(self, params, batch):
        x, t = batch
        dphi_dt, dc_dt = vmap(self.net_speed, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(jax.nn.relu(dphi_dt)) + jnp.mean(jax.nn.relu(dc_dt))
        # return jnp.mean(jax.nn.softplus(dphi_dt)) + jnp.mean(jax.nn.softplus(dc_dt))

    def loss_flux(self, params, batch):
        x, t = batch
        dphi_dx, dc_dx = vmap(self.net_nabla, in_axes=(None, 0, 0))(params, x, t)
        return jnp.mean(dphi_dx**2) + jnp.mean(dc_dx**2)

    @partial(jit, static_argnums=(0, 4))
    def compute_losses_and_grads(self, params, batch, eps, pde_name):
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
                )(params, batch_item, eps, pde_name)
                aux_vars.update(aux)
            else:
                loss_item, grad_item = jax.value_and_grad(loss_item_fn)(
                    params, batch_item
                )
            losses.append(loss_item)
            grads.append(grad_item)

        return jnp.array(losses), grads, aux_vars

    @partial(jit, static_argnums=(0, 4))
    def loss_fn(self, params, batch, eps, pde_name):
        losses, grads, aux_vars = self.compute_losses_and_grads(
            params, batch, eps, pde_name
        )

        weights = self.grad_norm_weights(grads)

        return jnp.sum(weights * losses), (losses, weights, aux_vars)

    def grad_norm_weights(self, grads: list, eps=1e-8):
        def tree_norm(pytree):
            squared_sum = sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(pytree))
            return jnp.sqrt(squared_sum)

        grad_norms = jnp.array([tree_norm(grad) for grad in grads])

        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        weights = jnp.mean(grad_norms) / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        weights = weights.at[1].set(weights[1] * 3)
        return jax.lax.stop_gradient(weights)
