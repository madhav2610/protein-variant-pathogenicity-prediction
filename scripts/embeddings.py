from transformers import AutoModel, AutoTokenizer
import torch

model_name = "facebook/esm2_t12_35M_UR50D"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name).to(device)

model.eval()
import pandas as pd

import numpy as np
import pickle
with open("/content/split.pkl",'rb') as f:
  train_df,test_df = pickle.load(f)
print(test_df.head())
print(train_df.head())


from google.colab import drive
import os


drive.mount('/content/drive')
SAVE_DIR = "/content/drive/MyDrive/variant_project"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device}")



def cache_originals(df, batch_size=4, max_length=1024):
    seq_to_positions = {}
    for seq, pos in zip(df["sequence"].tolist(), df["position"].tolist()):
        if seq not in seq_to_positions:
            seq_to_positions[seq] = set()
        seq_to_positions[seq].add(pos)

    unique_seqs = list(seq_to_positions.keys())
    total = len(unique_seqs)
    print(f"Caching {total} unique sequences...")
    seq_to_data = {}

    for i in range(0, total, batch_size):
        if i % 100 == 0:
            print(f"  {i}/{total}")
        batch = unique_seqs[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state
        mask   = inputs["attention_mask"]

        for j, seq in enumerate(batch):
            h       = hidden[j].cpu()
            m       = mask[j].cpu()
            seq_len = int(m.sum().item())
            m_exp   = m.unsqueeze(-1).float()
            mean    = (h * m_exp).sum(dim=0) / m_exp.sum().clamp(min=1)

            pos_embeddings = {}
            for pos in seq_to_positions[seq]:
                if pos < seq_len:
                    pos_embeddings[pos] = h[pos].numpy()
                else:
                    pos_embeddings[pos] = mean.numpy()

            seq_to_data[seq] = {
                "mean":     mean.numpy(),
                "pos_embs": pos_embeddings
            }

        torch.cuda.empty_cache()

    print("Caching done.")
    return seq_to_data


def get_strat2_strat3(df, seq_to_data, batch_size=4, max_length=1024,
                      start_from=0, checkpoint_every=5000):
    sequences = df["sequence"].tolist()
    positions = df["position"].tolist()
    mutations = df["mut"].tolist()

    # build mutated sequences — skip if seq > 1024
    mutated_sequences = []
    for seq, pos, mut in zip(sequences, positions, mutations):
        if len(seq) > 1024:
            mutated_sequences.append(None)  # will be skipped
        else:
            idx = pos - 1
            if idx < len(seq):
                mutated_sequences.append(seq[:idx] + mut + seq[idx+1:])
            else:
                mutated_sequences.append(seq)

    X_strat2 = []
    X_strat3 = []
    total = len(sequences)
    skipped = 0

    for i in range(start_from, total, batch_size):
        if i % 500 == 0:
            print(f"{i}/{total} | skipped so far: {skipped}")

        batch_orig = sequences[i:i+batch_size]
        batch_mut  = mutated_sequences[i:i+batch_size]
        batch_pos  = positions[i:i+batch_size]
        B = len(batch_orig)

        # filter out None (skipped long seqs) for forward pass
        valid_idx  = [j for j in range(B) if batch_mut[j] is not None]
        valid_muts = [batch_mut[j] for j in valid_idx]

        
        if valid_muts:
            inputs = tokenizer(valid_muts, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            mut_hidden = outputs.last_hidden_state
            mut_mask   = inputs["attention_mask"]

        valid_counter = 0
        for j in range(B):
            cached    = seq_to_data[batch_orig[j]]
            orig_mean = cached["mean"]

            if batch_mut[j] is None:
                X_strat2.append(np.zeros(480, dtype=np.float32))
                X_strat3.append(np.zeros(480, dtype=np.float32))
                skipped += 1
                continue

            orig_mean_t  = torch.tensor(orig_mean)
            mut_mask_cpu = mut_mask[valid_counter].cpu()
            mut_mask_exp = mut_mask_cpu.unsqueeze(-1).float()
            mut_sum      = (mut_hidden[valid_counter].cpu() * mut_mask_exp).sum(dim=0)
            mut_len      = mut_mask_exp.sum().clamp(min=1)
            mut_mean     = mut_sum / mut_len
            diff         = (mut_mean - orig_mean_t).numpy()

            emb_pos = cached["pos_embs"][batch_pos[j]]

            X_strat2.append(emb_pos)
            X_strat3.append(diff)
            valid_counter += 1

        torch.cuda.empty_cache()

        
        rows_done = i + B
        if rows_done % checkpoint_every < batch_size:
            s2 = np.array(X_strat2)
            s3 = np.array(X_strat3)
            np.save(f"{SAVE_DIR}/X_strat2_from{start_from}_at{rows_done}.npy", s2)
            np.save(f"{SAVE_DIR}/X_strat3_from{start_from}_at{rows_done}.npy", s3)
            print(f"✅ Saved to Drive at {rows_done} rows")

    return np.array(X_strat2), np.array(X_strat3)



START_FROM = 0

seq_to_data = cache_originals(test_df, batch_size=4)

print("\nStarting embedding extraction...")
X_strat2_test, X_strat3_test = get_strat2_strat3(
    test_df, seq_to_data,
    batch_size=4,
    start_from=START_FROM,
    checkpoint_every=5000
)

# final save
np.save(f"{SAVE_DIR}/X_strat2_from{START_FROM}_FINAL.npy", X_strat2_test)
np.save(f"{SAVE_DIR}/X_strat3_from{START_FROM}_FINAL.npy", X_strat3_test)
np.save(f"{SAVE_DIR}/y_train.npy", test_df["pathogenicity"].values_test)
print(f"Done! strat2: {X_strat2_test.shape}, strat3: {X_strat3_test.shape}")

np.save(f"{SAVE_DIR}/y_test.npy", test_df["pathogenicity"].values)
np.save(f"{SAVE_DIR}/X_strat2_test_FINAL.npy", X_strat2_test)
np.save(f"{SAVE_DIR}/X_strat3_test_FINAL.npy", X_strat3_test)