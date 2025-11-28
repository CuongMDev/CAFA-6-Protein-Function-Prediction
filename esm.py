import os
import re
import numpy as np
import torch
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer, T5EncoderModel, T5Tokenizer
from config import esm_model_name, t5_model_name, DEVICE, hf_cache, embedding_dim, train_emb_npy, train_seq_file, test_seq_unknown, test_emb_npy

# tokenizer = EsmTokenizer.from_pretrained(esm_model_name, cache_dir=hf_cache)
# model = EsmModel.from_pretrained(
#     esm_model_name, 
#     add_cross_attention=False,
#     is_decoder=False, 
#     cache_dir=hf_cache
# ).to(DEVICE)

tokenizer = T5Tokenizer.from_pretrained(t5_model_name, do_lower_case=False, cache_dir=hf_cache)
model = T5EncoderModel.from_pretrained(t5_model_name, cache_dir=hf_cache, use_safetensors=True).to(DEVICE)

# print sum model parameter, %6
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,}")

model.eval()

def get_embeddings(seq):
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]
    ids = tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(DEVICE)
    attention_mask = torch.tensor(ids['attention_mask']).to(DEVICE)

    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

    emb_0 = embedding_repr.last_hidden_state[0]
    emb_0_per_protein = emb_0.mean(dim=0)
    return emb_0_per_protein

def build_train_sequences(fasta_file):
    """
    Đọc file FASTA và trích chuỗi amino acid cho mỗi protein.
    Không chuẩn hóa hay chuyển sang số, chỉ trả về danh sách chuỗi amino acid.

    Returns:
        list: danh sách chuỗi amino acid (str)
    """
    sequences = []
    seq_lines = []

    with open(fasta_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # bỏ dòng rỗng
                continue

            if line.startswith(">"):
                # Append protein trước đó
                if seq_lines:
                    seq_str = "".join(seq_lines)
                    sequences.append(seq_str)
                seq_lines = []
            else:
                seq_lines.append(line)  # giữ nguyên chữ, không chuyển chữ hoa

        # Append protein cuối cùng
        if seq_lines:
            seq_str = "".join(seq_lines)
            sequences.append(seq_str)

    return sequences

def build_test_sequences(fasta_file):
    """
    Đọc file FASTA test và trích chuỗi amino acid cho mỗi protein.
    Chỉ trả về danh sách chuỗi amino acid, không chuẩn hóa hay chuyển sang số.

    Returns:
        list: danh sách chuỗi amino acid (str)
    """
    sequences = []

    with open(fasta_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('>'):
            # Bỏ header, chỉ quan tâm đến chuỗi amino acid
            i += 1
            sequence = ""
            while i < len(lines) and not lines[i].startswith('>'):
                sequence += lines[i].strip()
                i += 1
            sequences.append(sequence)
        else:
            i += 1

    return sequences

def embed_fasta_to_npy(sequences, output_file, embedding_dim=1280):
    """
    Đọc file FASTA, tạo embedding cho từng protein bằng get_embeddings, 
    và lưu embeddings ngay ra file .npy.
    
    Args:
        sequences (str): danh sách chuỗi amino acid.
        output_file (str): đường dẫn file .npy để lưu embeddings.
        get_embeddings (func): hàm get_embeddings(seq) trả về tensor embedding.
        embedding_dim (int): chiều embedding, mặc định 1280.
    """
    # Tạo mảng tạm (có thể thay bằng list nếu muốn incremental)
    num_sequences = len(sequences)
    embeds = np.zeros((num_sequences, embedding_dim))

    pbar = tqdm(total=num_sequences, desc="Embedding sequences", ncols=100)
    for i, seq in enumerate(sequences):
        embeds[i] = get_embeddings(seq).cpu().numpy()
        torch.cuda.empty_cache()
        
        # Update tqdm mỗi `batch_size` bước hoặc khi đến cuối
        if (i + 1) % 4000 == 0 or (i + 1) == num_sequences:
            pbar.update(4000)

    # Lưu embeddings ra file .npy
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, embeds)
    print(f"Saved {num_sequences} embeddings to {output_file}")

if __name__ == "__main__":
    # Đọc file FASTA, tạo embedding cho từng protein bằng get_embeddings, và lưu vào file .npy
    # embed_fasta_to_npy(build_train_sequences(train_seq_file), train_emb_npy, embedding_dim=1280)
    # embed_fasta_to_npy(build_test_sequences(test_seq_unknown), test_emb_npy, embedding_dim=1280)

    embed_fasta_to_npy(build_train_sequences(train_seq_file), train_emb_npy, embedding_dim=1024)
    embed_fasta_to_npy(build_test_sequences(test_emb_npy), test_emb_npy, embedding_dim=1024)