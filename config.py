import torch

data_dir = "Data/"
model_dir = "Model/"

esm_model_name = "facebook/esm2_t33_650M_UR50D"

train_seq_file = f"{data_dir}Train/train_sequences.fasta"
test_seq_file = f"{data_dir}Test/testsuperset.fasta"
terms_file = f"{data_dir}Train/train_terms.tsv"
taxonomy_file = f"{data_dir}Train/train_taxonomy.tsv"
ia_file = f"{data_dir}IA.tsv"
obo_file = f"{data_dir}Train/go-basic.obo"

train_emb_npy = f"{data_dir}embeddings/train_emb.npy"
test_emb_npy = f"{data_dir}embeddings/test_emb.npy"

hf_cache = "./hf_cache"


model_save_path = f"{model_dir}/hybrid_model.pth"

# model
embedding_dim=1280
learning_rate=0.001
weight_decay=1e-4

log_step=150
val_step=500

SCHEDULER_TYPE="cosine"
GAMMA=0.5
WARMUP_RATIO=0.1
EPOCHS = 5

BATCH_SIZE = 32
VAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
VAL_RATIO = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")