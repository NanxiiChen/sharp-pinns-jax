EPOCHS = 10000
N_SAMPLES = 50
LR = 5e-4
DECAY = 0.9
DECAY_EVERY = 1000
PAUSE_EVERY = 100

DOMAIN = [[-0.5, 0.5], [0, 1]]
DATA_PATH = "./data/pitting-active-1d.npz"
LOG_DIR = "./logs"
PREFIX = "pitting-active-1d"


NUM_LAYERS = 8
HIDDEN_DIM = 16
OUT_DIM = 2


ACT_NAME = "tanh"
ARCH_NAME = "mlp"
FOURIER_EMB = False


ALPHA_PHI = 1.03e-4
OMEGA_PHI = 1.76e7
MM = 7.94e-18
DD = 8.5e-10
AA = 5.35e7
LP = 1e-11
CSE = 1.
CLE = 5100/1.43e5


Lc = 1e-4
Tc = 1e5
AC_PRE_SCALE = 1.0
CH_PRE_SCALE = 1.0
