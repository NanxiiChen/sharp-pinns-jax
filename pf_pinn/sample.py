from jax import random
import jax.numpy as jnp
from pyDOE import lhs
import matplotlib.pyplot as plt


def mesh_flat(*args):
    return [
        coord.reshape(-1, 1) for coord in jnp.meshgrid(*args)
    ]


def lhs_sampling(mins, maxs, num):
    lb = jnp.array(mins)
    ub = jnp.array(maxs)
    if not len(lb) == len(ub):
        raise ValueError(f"mins and maxs should have the same length.")
    ret = lhs(len(lb), int(num)) * (ub - lb) + lb
    # return [ret[:, :-1], ret[:, -1:]]
    return ret

# in Torch
# def make_uniform_grid_data_transition(mins, maxs, num):
#     if not len(mins) == len(maxs) == len(num):
#         raise ValueError(f"mins, maxs, num should have the same length.")
    
#     each_col = [torch.linspace(mins[i], maxs[i], num[i], device=DEVICE)[1:-1]
#                 for i in range(len(mins))]
#     distances = [(maxs[i] - mins[i]) / (num[i] - 1) for i in range(len(mins))]
#     shift = [torch.tensor(np.random.uniform(-distances[i], distances[i], 1), device=DEVICE)
#              for i in range(len(distances))]
#     shift = torch.cat(shift, dim=0)
#     each_col = [each_col[i] + shift[i] for i in range(len(each_col))]

#     return torch.stack(torch.meshgrid(*each_col, indexing="ij"), axis=-1).reshape(-1, len(mins))


# In Jax
def shfted_grid(mins, maxs, num, key):
    if not len(mins) == len(maxs) == len(num):
        raise ValueError(f"mins, maxs, num should have the same length.")
    
    each_col = [jnp.linspace(mins[i], maxs[i], num[i])[1:-1]
                for i in range(len(mins))]
    distances = [(maxs[i] - mins[i]) / (num[i] - 1) for i in range(len(mins))]
    shift = [random.uniform(key, (1,), minval=-distances[i], maxval=distances[i])
             for i in range(len(distances))]
    shift = jnp.concatenate(shift, axis=0)
    each_col = [each_col[i] + shift[i] for i in range(len(each_col))]
    return jnp.stack(mesh_flat(*each_col), axis=-1).reshape(-1, len(mins))

if __name__ == "__main__":
    mins = [0, 0]
    maxs = [1, 1]
    num = 100
    # print(lhs_sampling(mins, maxs, num))
    data = lhs_sampling(mins, maxs, num)
    data2 = lhs_sampling(mins, maxs, num)
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(data2[:, 0], data2[:, 1])
    plt.show()
