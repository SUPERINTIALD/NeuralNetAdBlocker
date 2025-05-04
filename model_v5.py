from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, asdict
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader, random_split

print(f"MPS available: {torch.backends.mps.is_available()}")

#  Hyper‑parameters
MAX_LEN      = 32
EMBED_DIM    = 32
NUM_FILTERS  = 64
KERNEL_SIZES = (3, 4, 5)
BATCH_SIZE   = 128
EPOCHS       = 15
LR           = 3e-4
DATA_DIR     = pathlib.Path("dataset")
DEVICE       = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

#CHECKPOINT
@dataclass
class Checkpoint:
    vocab: dict
    state_dict: dict
    hyper_params: dict

def save_checkpoint(fp: str, model: nn.Module, vocab: dict, **hparams) -> None:
    ckpt = Checkpoint(vocab, model.state_dict(), hparams)
    torch.save(asdict(ckpt), fp)
    print(f"✓ checkpoint saved → {fp}")

def load_checkpoint(fp: str, device: str = "cpu"):
    data = torch.load(fp, map_location=device)
    vocab = data["vocab"]
    model = CharCNNClassifier(len(vocab) + 1).to(device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, vocab, data.get("hyper_params", {})

def read_lines(path: pathlib.Path) -> List[str]:
    with path.open(encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

# Load raw strings
ads      = read_lines(DATA_DIR / "is_ad_combines_final.txt")
non_ads  = read_lines(DATA_DIR / "non_ad.txt")
all_strs = ads + non_ads
all_lbls = [1] * len(ads) + [0] * len(non_ads)

# Build vocabulary (character‑level, lower‑cased)
chars      = sorted({c.lower() for c in "".join(all_strs)})
char2idx   = {ch: i + 1 for i, ch in enumerate(chars)}  # 0 = PAD/UNK
vocab_size = len(char2idx) + 1

def encode(s: str, max_len: int = MAX_LEN) -> torch.Tensor:
    idxs = [char2idx.get(c.lower(), 0) for c in s][:max_len]
    idxs += [0] * (max_len - len(idxs))
    return torch.tensor(idxs, dtype=torch.long)

class AdDataset(Dataset):
    def __init__(self, strings: List[str], labels: List[int]):
        self.strings, self.labels = strings, labels

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        return encode(self.strings[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)

#  Model definition – Character CNN

class CharCNNClassifier(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, EMBED_DIM, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(EMBED_DIM, NUM_FILTERS, k) for k in KERNEL_SIZES
        )
        self.classifier = nn.Sequential(
            nn.Linear(NUM_FILTERS * len(KERNEL_SIZES), 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L)
        x = self.embed(x).transpose(1, 2)                # (B, E, L)
        feats = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]
        x = torch.cat(feats, dim=1)
        return self.classifier(x).squeeze(1)             # (B,)

#  Training & evaluation helpers

def run_epoch(loader: DataLoader, model: nn.Module, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * y.size(0)
            preds = (torch.sigmoid(logits) >= 0.5)
            total_correct += (preds == y.bool()).sum().item()
            n += y.size(0)
    return total_loss / n, total_correct / n

def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            probs = torch.sigmoid(model(x)).cpu()
            y_true.extend(y.numpy())
            y_pred.extend((probs >= 0.9).long().numpy())
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = (torch.tensor(y_true) == torch.tensor(y_pred)).float().mean().item()
    return acc, prec, rec, f1


if __name__ == "__main__":
    dataset = AdDataset(all_strs, all_lbls)

    # 70/30 train‑test split (fixed seed for reproducibility)
    test_len  = int(len(dataset) * 0.30)
    train_len = len(dataset) - test_len
    train_ds, test_ds = random_split(dataset, [train_len, test_len],
                                     generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = CharCNNClassifier(vocab_size).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(train_dl, model, criterion, optimizer)
        print(f"epoch {epoch:2d} | train loss {tr_loss:.4f} | train acc {tr_acc:.3f}")

    # save final model
    save_checkpoint(
        "models/model_v5.5.pth",
        model,
        vocab=char2idx,
        max_len=MAX_LEN,
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
    )

    # evaluate on held‑out test set
    acc, prec, rec, f1 = evaluate(model, test_dl)
    print("\n🧪 Test‑set metrics")
    print(f"accuracy  : {acc:.3f}")
    print(f"precision : {prec:.3f}")
    print(f"recall    : {rec:.3f}")
    print(f"F1‑score  : {f1:.3f}")