EPOCHS = 50000
N_SAMPLES = 100
LR = 5e-4
DECAY = 0.9
DECAY_EVERY = 2000
PAUSE_EVERY = 100


DOMAIN = [[-1, 1], [0, 1]]
DATA_PATH = "./data/Allen_Cahn.mat"
LOG_DIR = "./logs"
PREFIX = "allen-cahn"


NUM_LAYERS = 4
HIDDEN_DIM = 40
OUT_DIM = 1


ACT_NAME = "tanh"
ARCH_NAME = "mlp"
FOURIER_EMB = True
