# train_val_test.py
# Minimal train/val/test loop for TinyCountModel (regression: total count)

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict


# ----------------- Utilities -----------------

class AvgMeter:
    """Tracks running average of a metric."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.n = 0
        self.avg = 0.0
    def update(self, val: float, count: int = 1):
        self.sum += float(val) * count
        self.n += count
        self.avg = self.sum / max(self.n, 1)


def mse_mae(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (MSE, MAE) tensors."""
    mse = nn.functional.mse_loss(pred, target)
    mae = nn.functional.l1_loss(pred, target)
    return mse, mae


def move_batch(batch, device):
    """
    Flexible batch mover:
    Accepts (x, y), (x, y, mask), longer tuples (we take first 3),
    or dicts with keys like: x/X/features, y/Y/label/target/total, mask/key_padding_mask.
    Ensures y is (B, 1).
    """
    # --- tuple / list ---
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 2:
            x = batch[0]
            y = batch[1]
            mask = batch[2] if len(batch) >= 3 else None
            x = x.to(device)
            y = y.to(device)
            if mask is not None:
                mask = mask.to(device)
            # Ensure y is (B,1)
            if y.dim() == 1:
                y = y.unsqueeze(1)
            return x, y, mask

    # --- dict ---
    if isinstance(batch, dict):
        # Try common key variants
        x = batch.get("x") or batch.get("X") or batch.get("features")
        y = (batch.get("y") or batch.get("Y") or batch.get("label") or
             batch.get("labels") or batch.get("target") or batch.get("total"))
        mask = (batch.get("mask") or batch.get("key_padding_mask") or
                batch.get("attention_mask") or batch.get("pad_mask"))
        if x is not None and y is not None:
            x = x.to(device)
            y = y.to(device)
            if mask is not None:
                mask = mask.to(device)
            if y.dim() == 1:
                y = y.unsqueeze(1)
            return x, y, mask

    # --- as a last resort: helpful debug print ---
    raise ValueError(
        f"Batch must be (x,y) or (x,y,mask) or dict with x/y[/mask], got type={type(batch)} "
        f"and structure: {repr(batch)[:200]}..."
    )

# ----------------- Epoch loops -----------------

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Dict[str, float]:
    model.train()
    mse_meter, mae_meter = AvgMeter(), AvgMeter()

    for batch in loader:
        x, y, mask = move_batch(batch, device)           # x: (B, Cin, L), y: (B, 1), mask: (B, L) or None
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            y_hat = model(x, mask)                       # (B, 1)
            loss_mse, loss_mae = mse_mae(y_hat, y)
            loss = loss_mse                              # optimize MSE; we also track MAE

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        mse_meter.update(loss_mse.item(), bs)
        mae_meter.update(loss_mae.item(), bs)

    return {"train_mse": mse_meter.avg, "train_mae": mae_meter.avg}


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device) -> Dict[str, float]:
    model.eval()
    mse_meter, mae_meter = AvgMeter(), AvgMeter()

    for batch in loader:
        x, y, mask = move_batch(batch, device)
        y_hat = model(x, mask)
        loss_mse, loss_mae = mse_mae(y_hat, y)
        bs = x.size(0)
        mse_meter.update(loss_mse.item(), bs)
        mae_meter.update(loss_mae.item(), bs)

    return {"mse": mse_meter.avg, "mae": mae_meter.avg}


# ----------------- Fit + Test -----------------

def fit(model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int = 15,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        use_amp: bool = True,
        device: Optional[torch.device] = None,
        ckpt_path: Optional[str] = "best_model.pt") -> Dict[str, float]:
    """
    Trains the model with AdamW on MSE, tracks MAE, keeps best val MSE checkpoint.
    Returns a dict with best validation metrics.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    best_val_mse = float("inf")
    best_metrics = {"val_mse": float("inf"), "val_mae": float("inf")}

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, scaler)
        if val_loader is not None:
            val_stats = evaluate(model, val_loader, device)
            val_mse, val_mae = val_stats["mse"], val_stats["mae"]

            # Save best checkpoint by validation MSE
            if val_mse < best_val_mse and ckpt_path is not None:
                best_val_mse = val_mse
                best_metrics = {"val_mse": val_mse, "val_mae": val_mae}
                torch.save({"model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "epoch": epoch}, ckpt_path)

            print(f"[Epoch {epoch:03d}] "
                  f"train MSE={train_stats['train_mse']:.4f}, MAE={train_stats['train_mae']:.4f} | "
                  f"val MSE={val_mse:.4f}, MAE={val_mae:.4f}")
        else:
            print(f"[Epoch {epoch:03d}] train MSE={train_stats['train_mse']:.4f}, "
                  f"MAE={train_stats['train_mae']:.4f}")

    return best_metrics


@torch.no_grad()
def test(model: nn.Module,
         test_loader: DataLoader,
         device: Optional[torch.device] = None,
         ckpt_path: Optional[str] = "best_model.pt") -> Dict[str, float]:
    """
    Loads best checkpoint (if provided) and evaluates on test set.
    Returns test metrics dict.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    stats = evaluate(model, test_loader, device)
    print(f"[Test] MSE={stats['mse']:.4f}, MAE={stats['mae']:.4f}")
    return stats


# ----------------- Example main (skeleton) -----------------
if __name__ == "__main__":
    from tiny_model import TinyCountModel  # import your TinyCountModel definition

    # Example dataset skeletons (replace with your real datasets)
    class DummySeqDataset(torch.utils.data.Dataset):
        """Returns (x, y, mask?) with shapes: x:(Cin,L), y:(1,), mask:(L,) or None."""
        def __init__(self, n=1024, cin=5, L=512, with_mask=False):
            super().__init__()
            self.n, self.cin, self.L, self.with_mask = n, cin, L, with_mask
        def __len__(self): return self.n
        def __getitem__(self, idx):
            x = torch.randn(self.cin, self.L)
            y = torch.rand(1) * 10.0  # e.g., total count in [0,10)
            if self.with_mask:
                mask = torch.ones(self.L, dtype=torch.bool)
                return x, y, mask
            return x, y

    # Build loaders
    train_ds = DummySeqDataset(n=2048, with_mask=False)
    val_ds   = DummySeqDataset(n=512,  with_mask=False)
    test_ds  = DummySeqDataset(n=512,  with_mask=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = TinyCountModel(cin=5, channels=64, num_heads=4, dropout=0.1)

    # Train + Validate
    best = fit(model, train_loader, val_loader,
               epochs=15, lr=3e-4, weight_decay=1e-4,
               use_amp=True, ckpt_path="best_model.pt")

    # Test
    test_stats = test(model, test_loader, ckpt_path="best_model.pt")
