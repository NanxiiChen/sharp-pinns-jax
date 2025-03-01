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
