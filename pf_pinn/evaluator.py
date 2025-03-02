import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap


def evaluate1D(pinn, params, batch, ref, **kwargs):
    x, t = batch
    # pred = pinn.net_u(params, x.reshape(-1, 1), t.reshape(-1, 1))[:, 0].reshape(x.shape)
    pred = vmap(lambda x, t: pinn.net_u(params, x, t)[0], in_axes=(0, 0))(
        x.reshape(-1, 1), t.reshape(-1, 1)
    ).reshape(x.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5),)
    ax = axes[0]
    vmin, vmax = kwargs.get("val_range", (-1, 1))
    xlim = kwargs.get("xlim", (0, 1))
    ylim = kwargs.get("ylim", (-1, 1))

    ax.pcolormesh(x, t, pred, cmap="coolwarm",
                  vmin=vmin, vmax=vmax)

    ax.set(xlabel="x", ylabel="t", title="Predicted",
           xlim=xlim, ylim=ylim)

    ax = axes[1]
    ax.pcolormesh(x, t, jnp.abs(pred - ref), cmap="coolwarm",
                  vmin=vmin, vmax=vmax)

    ax.set(xlabel="x", ylabel="t", title="Error",
           xlim=xlim, ylim=ylim)

    plt.tight_layout()

    error = jnp.mean((pred - ref) ** 2)
    return fig, error


def evaluate2D(pinn, params, mesh, ref_path, ts, **kwargs):
    fig, axes = plt.subplots(len(ts), 2, figsize=(10, 3*len(ts)))
    vmin, vmax = kwargs.get("val_range", (-1, 1))
    xlim = kwargs.get("xlim", (-0.5, 0.5))
    ylim = kwargs.get("ylim",(0, 0.5))
    error = 0
    for idx, tic in enumerate(ts):
        t = jnp.ones_like(mesh[:, 0:1]) * tic
        batch = (mesh, t)
        pred = vmap(lambda x, t: pinn.net_u(params, x, t)[0], in_axes=(0, 0))(
            mesh, t
        ).reshape(mesh.shape[0], 1)
        
        ax = axes[idx, 0]
        ax.scatter(mesh[:, 0], mesh[:, 1], c=pred[:, 0], cmap="coolwarm",)
        ax.set(xlabel="x", ylabel="y", title=f"t={tic}",
               xlim=xlim, ylim=ylim, aspect="equal")
        
        ref_sol = jnp.load(f"{ref_path}/sol-{tic:.3f}.npy")[:, 0:1]
        ax = axes[idx, 1]
        ax.scatter(mesh[:, 0], mesh[:, 1], c=jnp.abs(pred - ref_sol), cmap="coolwarm",)
        ax.set(xlabel="x", ylabel="y", title=f"t={tic}",
               xlim=xlim, ylim=ylim, aspect="equal")
        error += jnp.mean((pred - ref_sol) ** 2)
        
    plt.tight_layout()
    error /= len(ts)
    return fig, error / len(ts)
    