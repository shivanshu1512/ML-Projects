"""
model.py — Bidirectional LSTM model for time-series forecasting.
Built with PyTorch. Falls back to a simple LSTM if torch unavailable.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# PyTorch BiLSTM
# ---------------------------------------------------------------------------

if _HAS_TORCH:
    class BiLSTMForecaster(nn.Module):
        """
        Bidirectional LSTM for sequence forecasting.

        Architecture
        ------------
        Input → BiLSTM (n_layers) → Dropout → FC → Output
        """

        def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 64,
            n_layers: int = 2,
            dropout: float = 0.2,
            forecast_steps: int = 1,
            bidirectional: bool = True,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=bidirectional,
            )
            directions = 2 if bidirectional else 1
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size * directions, forecast_steps)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)          # (B, T, H*directions)
            out = self.dropout(out[:, -1, :])  # take last time step
            return self.fc(out)            # (B, forecast_steps)


def build_model(
    hidden_size: int = 64,
    n_layers: int = 2,
    dropout: float = 0.2,
    forecast_steps: int = 1,
    bidirectional: bool = True,
) -> "BiLSTMForecaster":
    if not _HAS_TORCH:
        raise ImportError("PyTorch required. Run: pip install torch")
    return BiLSTMForecaster(
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout=dropout,
        forecast_steps=forecast_steps,
        bidirectional=bidirectional,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: "BiLSTMForecaster",
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> list[float]:
    """Train *model* and return per-epoch training loss."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model.to(dev)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(dev)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(dev)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_t)
        scheduler.step(epoch_loss)
        losses.append(epoch_loss)
        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:>3}/{epochs} | Loss: {epoch_loss:.6f}")
    return losses


@torch.no_grad()
def predict(model: "BiLSTMForecaster", X_test: np.ndarray, device: str = "cpu") -> np.ndarray:
    import torch
    model.eval()
    dev = torch.device(device)
    X_t = torch.tensor(X_test, dtype=torch.float32).to(dev)
    return model(X_t).cpu().numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scaler=None) -> dict:
    """Return RMSE, MAE, MAPE, directional accuracy (in original price scale if scaler given)."""
    if scaler is not None:
        # inverse transform 1-step predictions
        y_true_orig = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_orig, y_pred_orig = y_true.flatten(), y_pred.flatten()

    rmse = np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))
    mae  = np.mean(np.abs(y_true_orig - y_pred_orig))
    mape = np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-8))) * 100
    # Directional accuracy: did the model predict the direction of change?
    if len(y_true_orig) > 1:
        dir_acc = np.mean(np.sign(np.diff(y_true_orig)) == np.sign(np.diff(y_pred_orig)))
    else:
        dir_acc = float("nan")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "Dir_Acc": dir_acc}
