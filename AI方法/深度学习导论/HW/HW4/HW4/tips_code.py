# python /root/HW4/HW4/tips_code.py
import time
import random
from dataclasses import dataclass
from typing import Any, Tuple, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchtext
torchtext.disable_torchtext_deprecation_warning()

from torchtext.datasets import SST2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


# =========================
# 0. Reproducibility
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 1. Robust sample parsing
# =========================
def split_text_label(sample):
    """
    支持三种情况：
    1) (text, label)
    2) (label, text)
    3) (text,)  —— test split 无 label
    返回: (text:str, label:int|None)
    """
    # case: (text,)
    if isinstance(sample, (list, tuple)) and len(sample) == 1:
        text = sample[0]
        if not isinstance(text, str):
            raise TypeError(f"Single-field sample but not text: {sample}")
        return text, None

    if not (isinstance(sample, (list, tuple)) and len(sample) == 2):
        raise ValueError(f"Unexpected sample format: {sample}")

    a, b = sample
    if isinstance(a, str) and not isinstance(b, str):
        text, label = a, b
    elif isinstance(b, str) and not isinstance(a, str):
        text, label = b, a
    elif isinstance(a, str) and isinstance(b, str):
        text, label = a, b
    else:
        raise TypeError(f"Cannot find text field in sample: {sample}")

    return text, int(label)


# =========================
# 2. Vocabulary
# =========================
def yield_tokens(data_iter: Iterable, tokenizer):
    for sample in data_iter:
        text, _ = split_text_label(sample)
        yield tokenizer(text)


def build_vocab(tokenizer, min_freq=2, max_size=50000):
    train_iter = SST2(split="train")
    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter, tokenizer),
        min_freq=min_freq,
        specials=["<pad>", "<unk>"],
        max_tokens=max_size,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def text_to_ids(text: str, tokenizer, vocab):
    return vocab(tokenizer(text))


# =========================
# 3. Batch + Collate
# =========================
@dataclass
class Batch:
    x: torch.Tensor
    lengths: torch.Tensor
    y: torch.Tensor | None


def collate_fn(batch, tokenizer, vocab, pad_idx: int, device):
    xs, ys = [], []

    for sample in batch:
        text, label = split_text_label(sample)
        ids = text_to_ids(text, tokenizer, vocab)
        if len(ids) == 0:
            ids = [vocab["<unk>"]]
        xs.append(torch.tensor(ids, dtype=torch.long))
        ys.append(label)

    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item())

    x_pad = torch.full((len(xs), max_len), pad_idx, dtype=torch.long)
    for i, x in enumerate(xs):
        x_pad[i, : x.size(0)] = x

    # test split: no labels
    if all(v is None for v in ys):
        y = None
    else:
        y = torch.tensor([int(v) for v in ys], dtype=torch.long).to(device)

    return Batch(
        x=x_pad.to(device),
        lengths=lengths.to(device),
        y=y,
    )


# =========================
# 4. Model
# =========================
class BiGRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 200,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 2)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.rnn(packed)

        if self.rnn.bidirectional:
            sent = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            sent = h_n[-1]

        sent = self.dropout(sent)
        return self.fc(sent)


# =========================
# 5. Train / Eval
# =========================
def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, criterion, save_pred_path=None):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    all_preds = []

    for batch in loader:
        logits = model(batch.x, batch.lengths)
        preds = logits.argmax(dim=1)

        if batch.y is None:
            all_preds.extend(preds.cpu().tolist())
            continue

        loss = criterion(logits, batch.y)
        bs = batch.y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, batch.y) * bs
        n += bs

    if n == 0 and save_pred_path is not None:
        with open(save_pred_path, "w", encoding="utf-8") as f:
            for p in all_preds:
                f.write(str(int(p)) + "\n")
        print(f"[Saved] test predictions -> {save_pred_path}")
        return None, None

    return total_loss / max(n, 1), total_acc / max(n, 1)


def train_one_epoch(model, loader, optimizer, criterion, clip=1.0):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch.x, batch.lengths)
        loss = criterion(logits, batch.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        bs = batch.y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, batch.y) * bs
        n += bs

    return total_loss / max(n, 1), total_acc / max(n, 1)


# =========================
# 6. Main
# =========================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    cfg = dict(
        batch_size=64,
        embed_dim=200,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5,
        lr=1e-3,
        epochs=10,
        min_freq=2,
        max_vocab=50000,
    )

    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab(tokenizer, cfg["min_freq"], cfg["max_vocab"])
    pad_idx = vocab["<pad>"]
    print("[Vocab size]", len(vocab), "pad_idx =", pad_idx)

    train_data = list(SST2(split="train"))
    dev_data = list(SST2(split="dev"))
    test_data = list(SST2(split="test"))

    train_loader = DataLoader(
        train_data,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, vocab, pad_idx, device),
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, vocab, pad_idx, device),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, vocab, pad_idx, device),
    )

    model = BiGRUClassifier(
        vocab_size=len(vocab),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pad_idx=pad_idx,
        bidirectional=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    best_dev = -1.0
    save_path = "best_sst2_bigru.pt"

    for ep in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        dv_loss, dv_acc = evaluate(model, dev_loader, criterion)
        if dv_acc > best_dev:
            best_dev = dv_acc
            torch.save(model.state_dict(), save_path)
        print(
            f"Epoch {ep:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"dev loss {dv_loss:.4f} acc {dv_acc:.4f} | "
            f"best {best_dev:.4f}"
        )

    model.load_state_dict(torch.load(save_path, map_location=device))
    te_loss, te_acc = evaluate(model, test_loader, criterion, save_pred_path="test_pred.txt")
    if te_loss is None:
        print("[TEST] no labels provided, predictions saved.")
    else:
        print(f"[TEST] loss={te_loss:.4f} acc={te_acc:.4f}")


if __name__ == "__main__":
    main()