# Central configuration for the tiny GPT training run.

D_MODEL = 192
N_HEADS = 8
D_FF = 768
N_LAYERS = 6
BLOCK_SIZE = 64
BATCH_SIZE = 16

N_STEPS = 4000
WARMUP_STEPS = 200
LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

EVAL_EVERY = 500
N_EVAL_STEPS = 2

CHECKPOINT_EVERY = 100
TOP_K = 30

LOG_EVERY = 50
PROGRESS_EVERY = 25

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/tinyshakespeare.txt"