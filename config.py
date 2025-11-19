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


SCHEDULER_TYPE="step"
STEP_SIZE=5
GAMMA=0.5
WARMUP_EPOCHS=3



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")