#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kalyanashisjana
"""


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import Descriptors
import selfies as sf

# === Load SMILES and encode SELFIES ===
df = pd.read_csv("carboxylic-lipinksi5.csv")
smiles_list = df["Smiles"].dropna().unique().tolist()
selfies_list = [sf.encoder(s) for s in smiles_list]

# Canonicalize training SMILES for novelty check
smiles_list_canonical = []
for s in smiles_list:
    mol = Chem.MolFromSmiles(s)
    if mol:
        smiles_list_canonical.append(Chem.MolToSmiles(mol, canonical=True))

# === Tokenizer ===
tokens = set()
for s in selfies_list:
    tokens.update(sf.split_selfies(s))

all_tokens = ["<START>"] + sorted(list(tokens)) + ["<PAD>", "<EOS>"]

char2idx = {tok: i for i, tok in enumerate(all_tokens)}
idx2char = {i: tok for i, tok in enumerate(all_tokens)}

vocab_size = len(all_tokens)
pad_idx = char2idx["<PAD>"]
eos_idx = char2idx["<EOS>"]

max_len = max(len(s) for s in selfies_list) + 20

def encode_selfies(s):
    toks = [char2idx["<START>"]] + [char2idx[tok] for tok in sf.split_selfies(s)] + [char2idx["<EOS>"]]
    while len(toks) < max_len:
        toks.append(char2idx["<PAD>"])
    return toks

def decode_selfies(indices):
    toks = []
    for idx in indices:
        if idx2char[idx] in ["<EOS>", "<PAD>"]:
            break
        toks.append(idx2char[idx])
    selfies_str = "".join(toks)
    return sf.decoder(selfies_str)

X = torch.tensor([encode_selfies(s) for s in selfies_list])

# === Custom scoring function ===
def calc_score(qed, logp, psa):
    # Scale parts to 0–1
    qed_score = qed  # already 0–1

    logp_target = 2  # ideal center
    logp_range = 1   # allowed deviation
    logp_score = max(0, 1 - abs(logp - logp_target) / logp_range)

    psa_target = 75
    psa_range = 25
    psa_score = max(0, 1 - abs(psa - psa_target) / psa_range)

    # Combine
    score = 0.5 * qed_score + 0.25 * logp_score + 0.25 * psa_score
    return score


# === Model ===
class SMILESRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, num_layers=2)
#        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        output, hidden = self.lstm(emb, hidden)
        logits = self.fc(output)
        return logits, hidden

model = SMILESRNN(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

# === Training ===
epochs = 200

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    inputs = X[:, :-1]
    targets = X[:, 1:]
    logits, _ = model(inputs)
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

# === Batch Sampling ===
def sample_single(model, temperature=0.6, max_length=100):
    model.eval()
    with torch.no_grad():
        input = torch.tensor([[char2idx["<START>"]]])
        hidden = None
        indices = []
        for _ in range(max_length):
            logits, hidden = model(input, hidden)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()
            if idx == eos_idx or idx == pad_idx:
                break
            indices.append(idx)
            input = torch.tensor([[idx]])
    return decode_selfies(indices)


# === Generate many ===
generated = set()

n_to_generate = 50
attempts = 0
max_attempts = n_to_generate * 10  # Allow some retries for invalids

while len(generated) < n_to_generate and attempts < max_attempts:
    new_smiles = sample_single(model, temperature=0.6)
    mol = Chem.MolFromSmiles(new_smiles)
    if mol:
        generated.add(new_smiles)
    attempts += 1

print(f"\nGenerated {len(generated)} valid unique SMILES.")

# === Compute descriptors for output ===
results = []
for s in generated:
    mol = Chem.MolFromSmiles(s)
    if not mol:
        continue

    # Canonicalize
    s_canonical = Chem.MolToSmiles(mol, canonical=True)
    is_novel = s_canonical not in smiles_list_canonical

    # Calculate descriptors
    logp = Descriptors.MolLogP(mol)
    qed = Descriptors.qed(mol)
    psa = Descriptors.TPSA(mol)
    hba = Descriptors.NumHAcceptors(mol)
    hbd = Descriptors.NumHDonors(mol)
    score = calc_score(qed, logp, psa)
    # Apply property filter
    if 0 < logp < 5 and psa < 140 and hba <= 10 and hbd <= 5:
        results.append({
            "Generated_SMILES": s,
            "Canonical_SMILES": s_canonical,
            "Novel": is_novel,
            "LogP": logp,
            "QED": qed,
            "TPSA": psa,
            "HAcceptors": hba,
            "HDonors": hbd,
            "Score": score
        })

# === Compute overall novelty score ===
num_novel = sum(1 for row in results if row["Novel"])
novelty_percent = 100 * num_novel / len(results) if results else 0
print(f"\nNovelty: {novelty_percent:.2f}% of generated molecules are new.")

# === Build DataFrame ===
df_gen = pd.DataFrame(results)

# Modern safe way to append summary row:
summary_df = pd.DataFrame([{
    "Generated_SMILES": "SUMMARY",
    "Canonical_SMILES": "",
    "Novel": f"{novelty_percent:.2f}%",
    "LogP": "",
    "QED": "",
    "TPSA": "",
    "HAcceptors": "",
    "HDonors": "",
    "Score": ""
}])

df_gen = pd.concat([df_gen, summary_df], ignore_index=True)

# === Save ===
df_gen.to_csv("generated_molecules.csv", index=False)
print("\n Saved generated molecules to generated_molecules.csv")
print(df_gen.head())

