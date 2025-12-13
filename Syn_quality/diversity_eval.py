import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
import math
from scipy.spatial.distance import pdist
from tqdm import tqdm
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import json
from datasets import load_dataset

# ===================== Parse command-line arguments =====================
parser = argparse.ArgumentParser(description="Evaluate dataset diversity with multiple metrics")
parser.add_argument("--model_name", type=str, required=True, help="Target model name or path")
parser.add_argument("--dataset", type=str, required=True, help="Path to txt dataset (one sample per line)")
parser.add_argument("--sample_size", type=int, default=200, help="Number of samples for Self-BLEU calculation")
parser.add_argument("--device", type=str, default="cuda", help="Device to run on (e.g., 'cuda', 'cuda:0', 'cpu')")
args = parser.parse_args()

# ===================== 1. Load data =====================
if args.dataset=="Official_bio":
    bio_corpus_path = "Synthetic_data/WMDP/bio/bio_remove_dataset.jsonl"
    with open(bio_corpus_path, "r", encoding="utf-8") as f:
        texts = [json.loads(line)["text"] for line in f if line.strip()]
elif args.dataset=="Official_cyber":
    ds = load_dataset("cais/wmdp-corpora", "cyber-forget-corpus", split="train")
    texts = list(ds["text"])
elif args.dataset=="Official_forget01":
    ds = load_dataset("locuslab/TOFU", "forget01")
    texts = []
    train_dataset = ds["train"]
    texts.append([f"{question}: {answer}" for question, answer in zip(train_dataset["question"], train_dataset["answer"])])
    texts = texts[0]
elif args.dataset=="Official_forget05":
    ds = load_dataset("locuslab/TOFU", "forget05")
    texts = []
    train_dataset = ds["train"]
    texts.append([f"{question}: {answer}" for question, answer in zip(train_dataset["question"], train_dataset["answer"])])
    texts = texts[0]
elif args.dataset=="Official_forget10":
    ds = load_dataset("locuslab/TOFU", "forget10")
    texts = []
    train_dataset = ds["train"]
    texts.append([f"{question}: {answer}" for question, answer in zip(train_dataset["question"], train_dataset["answer"])])
    texts = texts[0]
else:
    with open(args.dataset, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
print(f"Loaded {len(texts)} samples from {args.dataset}")

# ===================== 2. Load model and tokenizer =====================
if args.model_name=="muse-bench/MUSE-books_target":
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModel.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(args.device).eval()

# ===================== 3. Generate semantic embeddings =====================
def get_embedding(text, pool="mean"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(args.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state
    if pool == "mean":
        emb = hidden_states.mean(dim=1).squeeze().cpu().numpy()
    else:
        emb = hidden_states[:, -1, :].squeeze().cpu().numpy()
    return emb

embeddings = np.array([get_embedding(t) for t in tqdm(texts, desc="Extracting embeddings")])

# ===================== Visualize embeddings (PCA) =====================
# Research-style configuration
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.titlesize": 40,
    "axes.labelsize": 35,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 34,
    "figure.dpi": 150,
    "axes.linewidth": 1.2,
})

# PCA dimensionality reduction
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot
plt.figure(figsize=(7, 6))
plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    alpha=0.5,
    s=40,                   # Marker size
    c="#2570b4",            # Blue tone commonly used in research figures
    edgecolor="none"
)
plt.title("PCA Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
# Use the dataset filename without extension
dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
# Save high-resolution figure
plt.tight_layout()
plt.savefig(f"Syn_quality/Figures/Diversity/{dataset_name}.pdf", dpi=300)   # PDF format
plt.show()

# ===================== Remote-Clique =====================
def remote_clique_score(embeddings):
    distances = pdist(embeddings, metric="cosine")
    return np.mean(distances)

rc_score = remote_clique_score(embeddings)
print(f"Remote-Clique Score: {rc_score:.4f}")
