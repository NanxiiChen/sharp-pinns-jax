from matplotlib import RcParams
import matplotlib
import jax

from pf_pinn.arch import MLP, ModifiedMLP
from pf_pinn.evaluator import evaluate1D, evaluate2D
from pf_pinn.metrics import MetricsTracker
from pf_pinn.sample import mesh_flat, lhs_sampling, shfted_grid
from pf_pinn.causal import CausalWeightor, update_causal_eps
from pf_pinn.utils import StaggerSwitch

jax.config.update("jax_default_matmul_precision", "high")


matplotlib.use("Agg")
RcParams.update({
    "font.size": 16,
    "font.sans-serif": ["Arial", "Helvetica"],
    "font.family": "sans-serif",
    "xtick.direction": "in",
    "ytick.direction": "in",
})
