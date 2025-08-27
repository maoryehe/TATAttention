# src/main.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.makeLoaders import make_loaders
from src.model.TotalCountModel import TotalCountModel

def _get_xy(batch, device):
    # Accepts (x,y) or (x,y,mask) or dict with x/X/features and y/Y/label/target/total
    import torch

    if isinstance(batch, (tuple, list)):
        x = batch[0]; y = batch[1]
    elif isinstance(batch, dict):
        def pick(d, keys):
            for k in keys:
                if k in d:
                    return d[k]
            return None
        x = pick(batch, ["x", "X", "features"])
        y = pick(batch, ["y", "Y", "label", "labels", "target", "total", "count", "counts"])
        if x is None or y is None:
            raise ValueError(f"Dict batch missing x/y keys. Got keys={list(batch.keys())}")
    else:
        raise ValueError(f"Unexpected batch type: {type(batch)}")

    if not torch.is_tensor(x): x = torch.as_tensor(x)
    if not torch.is_tensor(y): y = torch.as_tensor(y)

    x = x.to(device)
    y = y.to(device)

    # Ensure y shape is (B,1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    return x, y


def main():
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npz_path = os.path.join(project_root, "data", "processed_counts_onehot_L600.npz")
    ckpt_path = os.path.join(project_root, "checkpoints", "best_totalcount.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    # Loaders (your make_loaders takes only the path — keep it simple)
    train_loader, val_loader, test_loader = make_loaders(npz_path)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TotalCountModel(cin=4, channels=64, num_heads=4, dropout=0.1).to(device)

    # Optim & loss
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # Train/Val
    best_val = float("inf")
    EPOCHS = 10
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0; n = 0
        for batch in train_loader:
            x, y = _get_xy(batch, device)
            optim.zero_grad(set_to_none=True)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_mse = running / max(n, 1)

        # validation
        model.eval()
        running = 0.0; n = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = _get_xy(batch, device)
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                running += loss.item() * x.size(0)
                n += x.size(0)
        val_mse = running / max(n, 1)

        print(f"[Epoch {epoch:02d}] train MSE={train_mse:.4f} | val MSE={val_mse:.4f}")

        # save best
        if val_mse < best_val:
            best_val = val_mse
            torch.save({"model_state": model.state_dict()}, ckpt_path)

    # Test
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    running = 0.0; n = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = _get_xy(batch, device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            running += loss.item() * x.size(0)
            n += x.size(0)
    test_mse = running / max(n, 1)
    print(f"[Test] MSE={test_mse:.4f}")

if __name__ == "__main__":
    main()
