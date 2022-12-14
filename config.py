import torch
import random
import numpy as np
import transformers

from transformers import BertTokenizer

RANDOM_SEED = 123
NUM_EPOCHS = 2
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
DEVICE = torch.device("cuda")
LEARNING_RATE = 5e-5
MAX_LEN = 300  # max 512 (strongly affects GPU memory consumption)
ADAM_EPSILON = 1e-5
PRETAINED_MODEL = "Rostlab/prot_bert_bfd_localization"
TOKENIZER = BertTokenizer.from_pretrained(
        PRETAINED_MODEL, do_lower_case=True)
NUM_CLASSES = 11
GRADIENT_ACCUMULATION_STEPS = 8
DROPOUT_RATE = 0.3
HIDDEN_SIZE = 768  # size of BERT hidden layer
NUM_WORKERS = 8
MODEL_PATH = "model/"
DATA_PATH = "data/"

### reproducibility
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
