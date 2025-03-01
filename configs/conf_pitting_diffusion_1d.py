EPOCHS = 50000
N_SAMPLES = 100
LR = 5e-4
DECAY = 0.9
DECAY_EVERY = 2000
PAUSE_EVERY = 100


DOMAIN = [[-0.5, 0.5], [0, 1]]
DATA_PATH = "./data/pitting-diffusion-1d.npz"
LOG_DIR = "./logs"
PREFIX = "pitting-diffusion-1d"


NUM_LAYERS = 6
HIDDEN_DIM = 64
OUT_DIM = 2


ACT_NAME = "tanh"
ARCH_NAME = "mlp"
FOURIER_EMB = False


ALPHA_PHI = 1.03e-4
OMEGA_PHI = 1.76e7
MM = 7.94e-18
DD = 8.5e-10
AA = 5.35e7
LP = 2.0
CSE = 1.
CLE = 5100/1.43e5


Lc = 1e-4
Tc = 1e2
AC_PRE_SCALE = 1e9
CH_PRE_SCALE = 1e3
