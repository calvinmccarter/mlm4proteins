"""PyTorch port of the Mason/Tz_her2 trastuzumab CDR-H3 binding-affinity CNN.

Original architecture (Mason et al. 2021, Nat Biomed Eng; Tz_her2 repo):

    Conv1D(filters=400, kernel=5, stride=1, padding="same", relu)
    Dropout(0.2)
    MaxPool1D(pool=2, stride=1)
    Flatten
    Dense(300, relu)
    Dense(1, sigmoid)

Loss: binary cross-entropy. Optimizer: Adam, LR 7.5e-5. Batch size 16.
Training: 10 epochs with early stopping (patience 5) on val_loss.

The dataset is the HER2-affinity-large CSVs from the Tz_her2 repo:
``data/affinity_data/her2/her2_aff_large/processed/remove_overlap/random_split/0.7_0.15_0.15/{train,val,test}.csv``,
each with columns ``seq`` (10-aa H3) and ``label`` (binary binder/non-binder).

Target metrics on the held-out test set: ROC AUC ≈ 0.91, PR AUC ≈ 0.83.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mmmlm.constants import AA_TO_IDX, NUM_AMINO_ACIDS


def encode_sequences(seqs: Sequence[str], length: int = 10) -> np.ndarray:
    """One-hot encode sequences into a (N, length, 20) float array."""
    n = len(seqs)
    out = np.zeros((n, length, NUM_AMINO_ACIDS), dtype=np.float32)
    for i, s in enumerate(seqs):
        if len(s) != length:
            raise ValueError(f"sequence {i} has length {len(s)}, expected {length}")
        for j, aa in enumerate(s):
            if aa not in AA_TO_IDX:
                raise ValueError(f"unknown AA {aa!r} in sequence {i}")
            out[i, j, AA_TO_IDX[aa]] = 1.0
    return out


class TrastuzumabCNN(nn.Module):
    """Mason-style CNN for 10-aa H3 binder classification."""

    def __init__(self, seq_length: int = 10, n_amino: int = NUM_AMINO_ACIDS):
        super().__init__()
        self.seq_length = seq_length
        # Conv1d expects (B, C, L); we'll permute the (B, L, C) input.
        self.conv = nn.Conv1d(
            in_channels=n_amino, out_channels=400, kernel_size=5, stride=1, padding="same"
        )
        self.dropout = nn.Dropout(0.2)
        # MaxPool1d: pool=2, stride=1 → output length = L - 1
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        pooled_len = seq_length - 1
        self.fc1 = nn.Linear(400 * pooled_len, 300)
        self.fc2 = nn.Linear(300, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is (B, L, 20). Returns sigmoid output of shape (B,)."""
        x = x.transpose(1, 2)        # (B, 20, L)
        x = F.relu(self.conv(x))     # (B, 400, L)
        x = self.dropout(x)
        x = self.pool(x)             # (B, 400, L-1)
        x = x.flatten(1)             # (B, 400*(L-1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)
        return torch.sigmoid(x)


# ----- training -----

def _load_split(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    X = encode_sequences(df["seq"].tolist())
    y = df["label"].astype(np.float32).values
    return X, y


def train_oracle(
    data_dir: Path,
    out_path: Path,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 7.5e-5,
    patience: int = 5,
    device: str | torch.device | None = None,
    seed: int = 0,
) -> dict:
    """Train the CNN on the HER2-affinity-large data and save the checkpoint.

    Returns a dict of test ROC AUC / PR AUC and best-epoch info.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    X_tr, y_tr = _load_split(data_dir / "train.csv")
    X_va, y_va = _load_split(data_dir / "val.csv")
    X_te, y_te = _load_split(data_dir / "test.csv")

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=2)

    model = TrastuzumabCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            yhat = model(xb)
            loss = bce(yhat, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running_loss / n_seen

        model.eval()
        with torch.inference_mode():
            v_loss_sum, v_n = 0.0, 0
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                yhat = model(xb)
                v_loss_sum += bce(yhat, yb).item() * xb.size(0)
                v_n += xb.size(0)
            val_loss = v_loss_sum / v_n

        improved = val_loss < best_val_loss - 1e-6
        marker = " *" if improved else ""
        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{marker}")
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test eval
    model.eval()
    with torch.inference_mode():
        Xt = torch.from_numpy(X_te).to(device)
        # Predict in chunks to keep memory reasonable.
        preds = []
        for i in range(0, Xt.size(0), 1024):
            preds.append(model(Xt[i:i + 1024]).cpu().numpy())
        y_pred = np.concatenate(preds)

    roc = roc_auc_score(y_te, y_pred)
    prc = average_precision_score(y_te, y_pred)
    print(f"test ROC AUC = {roc:.4f}   PR AUC = {prc:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "test_roc_auc": roc, "test_pr_auc": prc}, out_path)
    return {"test_roc_auc": float(roc), "test_pr_auc": float(prc), "best_val_loss": float(best_val_loss)}


def load_oracle(checkpoint_path: Path, device: str | torch.device | None = None) -> TrastuzumabCNN:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TrastuzumabCNN().to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def predict(model: TrastuzumabCNN, sequences: Sequence[str], batch_size: int = 1024) -> np.ndarray:
    """Predict binder probability for a batch of 10-aa H3 sequences."""
    device = next(model.parameters()).device
    X = encode_sequences(sequences)
    out = []
    for i in range(0, X.shape[0], batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Directory with train.csv / val.csv / test.csv")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output checkpoint path (.pt)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_oracle(args.data_dir, args.out, epochs=args.epochs, seed=args.seed)
