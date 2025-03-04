import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable
import matplotlib.pyplot as plt



# from jax import config
# config.update("jax_disable_jit", True)


class CausalWeightor:
    def __init__(self, num_chunks: int, t_range: tuple):
        self.causal_configs = {
            "eps": 1e-3,
            "step_size": 10,
            "max_last_weight": 0.99,
            "min_mean_weight": 0.5,
            "max_eps": 1e-3
        }
        self.num_chunks = num_chunks
        self.t_range = t_range
        self.bins = jnp.linspace(t_range[0], t_range[1], num_chunks+1)
    
    
    # def _update_causal_configs(self, causal_weights):

    #     if causal_weights.min() > self.causal_configs["max_last_weight"] \
    #         and self.causal_configs["eps"] < self.causal_configs["max_eps"]:
    #             self.causal_configs["eps"] *= self.causal_configs["step_size"]
    #             # print(f"Inc. eps to {self.causal_configs['eps']}")
        
    #     if jnp.mean(causal_weights) < self.causal_configs["min_mean_weight"]:
    #         self.causal_configs["eps"] /= self.causal_configs["step_size"]
    #         # print(f"Dec. eps to {self.causal_configs['eps']}")
    
    def _update_causal_configs(self):
        eps_init = self.causal_configs["eps"]
        causal_weights = self.causal_weights
        
        cond1 = (causal_weights.min() > self.causal_configs["max_last_weight"]) & (eps_init < self.causal_configs["max_eps"])
        eps_after_cond1 = jnp.where(cond1, eps_init * self.causal_configs["step_size"], eps_init)

        cond2 = jnp.mean(causal_weights) < self.causal_configs["min_mean_weight"]
        new_eps = jnp.where(cond2, eps_after_cond1 / self.causal_configs["step_size"], eps_after_cond1)

        self.causal_configs.update({"eps": new_eps})


    def _split_t(self, t: jnp.array, ):
        num = self.num_chunks
        bins = self.bins
        t_idx = jnp.digitize(t, bins) - 1

        return [jnp.where(t_idx == int(i))[0] for i in range(num)]

    def _compute_causal_weight(self, loss_chunks: jnp.array):
        eps = self.causal_configs["eps"]
        weights = jnp.zeros(self.num_chunks)
        for chunk_id in range(self.num_chunks):
            if chunk_id == 0:
                weights = weights.at[chunk_id].set(1.0)
            else:
                weights = weights.at[chunk_id].set(jnp.exp(
                    -eps * jnp.sum(loss_chunks[:chunk_id])
                ))
        weights = jax.lax.stop_gradient(weights)
        # self.causal_weights = jax.device_get(weights)
        return weights

    # @partial(jax.jit, static_argnums=(0,))
    # def compute_causal_loss(self, residuals: jnp.array, t: jnp.array):
    #     # loss_chunks = jnp.zeros(self.num_chunks)
    #     # for chunk_idx, data_idx in enumerate(indices):
    #     #     loss_chunks[chunk_idx] = jnp.mean(residuals[data_idx] ** 2)
    #     indices = self._split_t(t)
    #     loss_chunks = jnp.array([jnp.mean(residuals[data_idx] ** 2) for data_idx in indices])
    #     causal_weights = self._compute_causal_weight(loss_chunks)
    #     causal_loss = jnp.dot(loss_chunks, causal_weights)
    #     self.caual_loss = causal_loss
    #     return causal_loss
    
    def compute_causal_loss(self, residuals: jnp.array, t: jnp.array):
        # 1. 计算每个样本所在的分段编号
        t_idx = jnp.digitize(t.flatten(), self.bins) - 1
        

        sum_residuals_sq = jax.ops.segment_sum(residuals**2, t_idx, num_segments=self.num_chunks)
        count_residuals = jax.ops.segment_sum(jnp.ones_like(residuals), t_idx, num_segments=self.num_chunks)
        loss_chunks = sum_residuals_sq / (count_residuals + 1e-12)
        
        causal_weights = self._compute_causal_weight(loss_chunks)
        causal_loss = jnp.dot(loss_chunks, causal_weights)
        # self.aux_vars["causal_weights"] = jax.device_get(causal_weights)
        # self.aux_vars["loss_chunks"] = jax.device_get(loss_chunks)
        return causal_loss
    
    
    def plot_causal_info(self, pde_name, causal_weights, loss_chunks):
        
        bins = (self.bins[1:] + self.bins[:-1]) / 2
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        ax = axes[0]
        ax.plot(bins, causal_weights, marker="o")
        ax.set(xlabel="Time chunks", ylabel="Causal weights", title=f"Causal weights for {pde_name}")
        
        ax = axes[1]
        ax.plot(bins, loss_chunks, marker="o")
        ax.set(xlabel="Time chunks", ylabel="Causal loss", title=f"Causal loss for {pde_name}")
        
        return fig
        
        

        
    

if __name__ == "__main__":
    t = jnp.array([0.0, 0.1, 0.7, 0.9, 0.3, 0.5, 1.0]).reshape(-1, 1)
    t_range = (0.0, 1.0)
    num = 4
    weightor = CausalWeightor(num, t_range)
    print(weightor._split_t(t))