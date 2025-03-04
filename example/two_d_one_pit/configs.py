EPOCHS = 100000
N_SAMPLES = 20
LR = 5e-4
DECAY = 0.9
DECAY_EVERY = 100
PAUSE_EVERY = 100


DOMAIN = [[-0.5, 0.5], [0, 0.5], [0, 1]]
DATA_PATH = "./data/2d-1pit/"
LOG_DIR = "/root/tf-logs"
PREFIX = "2d-1pit"
TS = [0.000, 1.534, 5.118, 9.726]

NUM_LAYERS = 6
HIDDEN_DIM = 128
OUT_DIM = 2


ACT_NAME = "gelu"
ARCH_NAME = "modified_mlp"
FOURIER_EMB = True



ALPHA_PHI = 1.03e-4
OMEGA_PHI = 1.76e7
MM = 7.94e-18
DD = 8.5e-10
AA = 5.35e7
LP = 2.0
CSE = 1.
CLE = 5100/1.43e5


Lc = 1e-4
Tc = 10.0
AC_PRE_SCALE = 1e6
CH_PRE_SCALE = 1e0
