import torch

data_dir = "Data/"
model_dir = "Model/"

train_seq_file = f"{data_dir}Train/train_sequences.fasta"
test_seq_file = f"{data_dir}Test/testsuperset.fasta"
terms_file = f"{data_dir}Train/train_terms.tsv"
taxonomy_file = f"{data_dir}Train/train_taxonomy.tsv"
ia_file = f"{data_dir}IA.tsv"
obo_file = f"{data_dir}Train/go-basic.obo"


model_save_path = f"{model_dir}/hybrid_model.pth"

# model
lstm_hidden=256
lstm_layers=4
linear_hidden_dim=256
classifier_hidden_dim=256
learning_rate=0.001
weight_decay=1e-4

log_step=300
val_step=6000

SCHEDULER_TYPE="cosine"
GAMMA=0.5
WARMUP_RATIO=0.1
EPOCHS = 5

BATCH_SIZE = 2
VAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
VAL_RATIO = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")