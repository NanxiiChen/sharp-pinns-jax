EPOCHS = 100000
N_SAMPLES = 50
LR = 1e-3
DECAY = 0.9
DECAY_EVERY = 5000
PAUSE_EVERY = 200


DOMAIN = [[-0.5, 0.5], [0, 0.5], [0, 1]]
DATA_PATH = "./data/2d-1pit/"
LOG_DIR = "/root/tf-logs"
PREFIX = "2d-pit"


NUM_LAYERS = 8
HIDDEN_DIM = 64
OUT_DIM = 2


ACT_NAME = "gelu"
ARCH_NAME = "modified_mlp"
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
Tc = 1e1
AC_PRE_SCALE = 1e6
CH_PRE_SCALE = 1e0
