class Config:
    EPOCHS = 100000
    N_SAMPLES = 25
    ADAPTIVE_SAMPLES = 4000
    ADAPTIVE_BASE_RATE = 10
    LR = 5e-4
    DECAY = 0.9
    DECAY_EVERY = 200
    STAGGER_PERIOD = 25


    DOMAIN = [[-0.5, 0.5], [0, 0.5], [0, 1]]
    DATA_PATH = "./data/2d-2pits/"
    LOG_DIR = "/root/tf-logs"
    PREFIX = "2d-2pits"
    TS = [0.000, 2.273, 4.803, 9.528]

    NUM_LAYERS = 6
    HIDDEN_DIM = 200
    OUT_DIM = 2


    ACT_NAME = "gelu"
    ARCH_NAME = "modified_mlp"
    FOURIER_EMB = True
    CAUSAL_WEIGHT = True

    ALPHA_PHI = 1.03e-4
    OMEGA_PHI = 1.76e7
    MM = 7.94e-18
    DD = 8.5e-10
    AA = 5.35e7
    LP = 2.0
    CSE = 1.0
    CLE = 5100 / 1.43e5


    Lc = 1e-4
    Tc = 10.0
    AC_PRE_SCALE = 1e6
    CH_PRE_SCALE = 1e0


    CAUSAL_CONFIGS = {
        "ac_eps": 1e-4,
        "ch_eps": 1e-4,
        "step_size": 10,
        "max_last_weight": 0.99,
        "min_mean_weight": 0.5,
        "max_eps": 1e0,
        "chunks": 20,
    }


# if __name__ == "__main__":
#     for key, value in Config.__dict__.items():
#         # 将所以的item作为全局变量，key = value 的形式
#         if not key.startswith("__"):
#             globals()[key] = value