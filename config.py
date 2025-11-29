import torch

data_dir = "Data/"
model_dir = "Model/"
test_dir = f"{data_dir}Test/"

k_folds = 5

esm_model_name = "facebook/esm2_t33_650M_UR50D"
t5_model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"

train_seq_file = f"{data_dir}Train/train_sequences.fasta"
test_seq_file = f"{test_dir}testsuperset.fasta"
terms_file = f"{data_dir}Train/train_terms.tsv"
taxonomy_file = f"{data_dir}Train/train_taxonomy.tsv"
ia_file = f"{data_dir}IA.tsv"
obo_file = f"{data_dir}Train/go-basic.obo"

presubmit_file = f"{data_dir}archive/submission.tsv"

submit_known = f"{test_dir}submit_already_known.tsv"

esm_train_emb_npy = f"{data_dir}embeddings/esm_train_emb.npy"
esm_test_emb_npy = f"{data_dir}embeddings/esm_test_emb.npy"

t5_train_emb_npy = f"{data_dir}embeddings/t5_train_emb.npy"
t5_test_emb_npy = f"{data_dir}embeddings/t5_test_emb.npy"

hf_cache = "./hf_cache"

top_k = [3000, 1000, 500]
top_k_type = ['P', 'F', 'C']
NUM_CLASSES = sum(top_k)

model_save_path = f"{model_dir}/hybrid_model.pth"

# model
embedding_dim = 1280 + 1024
learning_rate=1e-3
weight_decay=1e-1

log_step=150
val_step=300

SCHEDULER_TYPE="cosine"
GAMMA=0.5
WARMUP_RATIO=0.1
EPOCHS = 20

BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 1
VAL_RATIO = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")