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
