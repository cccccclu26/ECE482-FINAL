"""
LSTM Model - Baseline neural network for stock direction prediction.

Uses PyTorch LSTM to predict whether price will be higher after N trading days.
Serves as a data-driven baseline to compare against LLM predictions.

Features per day (input sequence):
  - Daily return (close-to-close %)
  - RSI(14)
  - Price vs EMA25 %
  - Price vs EMA50 %
  - Price vs EMA100 %
  - EMA alignment score
  - Volume ratio
  Total: 7 features x LOOKBACK_SEQUENCE days
"""
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

import config
from technical import compute_ema, compute_rsi


# ============================================================
# Dataset
# ============================================================

class StockSequenceDataset(Dataset):
    """PyTorch dataset: sequences of daily features -> binary label."""

    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# ============================================================
# LSTM Network
# ============================================================

class StockLSTM(nn.Module):
    """LSTM for binary stock direction prediction."""

    def __init__(self, input_size=7, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


# ============================================================
# Feature Engineering
# ============================================================

def build_daily_features(price_df):
    """
    Build daily feature matrix from OHLCV data.

    Returns:
        DataFrame with 7 features per row, aligned with price_df index
    """
    close = price_df["Close"]
    volume = price_df["Volume"] if "Volume" in price_df.columns else pd.Series(0, index=price_df.index)

    ema25 = compute_ema(close, 25)
    ema50 = compute_ema(close, 50)
    ema100 = compute_ema(close, 100)
    rsi = compute_rsi(close, 14)

    daily_return = close.pct_change() * 100
    price_vs_ema25 = (close - ema25) / ema25 * 100
    price_vs_ema50 = (close - ema50) / ema50 * 100
    price_vs_ema100 = (close - ema100) / ema100 * 100

    # EMA alignment score
    ema100_uptrend = (ema100 > ema100.shift(20)).astype(float)
    ema50_above_100 = (ema50 > ema100).astype(float)
    ema25_above_50 = (ema25 > ema50).astype(float)
    price_above_25 = (close > ema25).astype(float)
    alignment = (
        price_above_25 + ema25_above_50 + ema50_above_100 + ema100_uptrend
        - (1 - price_above_25) - (1 - ema25_above_50)
        - (1 - ema50_above_100) - (1 - ema100_uptrend)
    )

    # Volume ratio
    vol_avg = volume.rolling(20).mean()
    vol_ratio = volume / vol_avg
    vol_ratio = vol_ratio.fillna(1.0)

    features = pd.DataFrame({
        "daily_return": daily_return,
        "rsi": rsi,
        "price_vs_ema25": price_vs_ema25,
        "price_vs_ema50": price_vs_ema50,
        "price_vs_ema100": price_vs_ema100,
        "ema_alignment": alignment,
        "volume_ratio": vol_ratio,
    }, index=price_df.index)

    return features.dropna()


def create_sequences(features_df, price_df, seq_len, horizon):
    """
    Create input sequences and binary labels.

    Args:
        features_df: Daily feature DataFrame
        price_df: Price DataFrame (for computing forward returns)
        seq_len: Number of past days in each sequence
        horizon: Forward days for label computation

    Returns:
        sequences: np.array (N, seq_len, n_features)
        labels: np.array (N,) binary 0/1
        dates: list of prediction dates
    """
    close = price_df["Close"]
    feature_vals = features_df.values
    feature_dates = features_df.index

    sequences = []
    labels = []
    dates = []

    for i in range(seq_len, len(feature_vals)):
        current_date = feature_dates[i]

        # Find the price horizon days forward
        future_idx = close.index.get_indexer([current_date], method="nearest")[0]
        target_idx = future_idx + horizon
        if target_idx >= len(close):
            continue

        current_price = close.iloc[future_idx]
        future_price = close.iloc[target_idx]
        label = 1.0 if future_price > current_price else 0.0

        seq = feature_vals[i - seq_len:i]
        sequences.append(seq)
        labels.append(label)
        dates.append(current_date)

    return np.array(sequences), np.array(labels), dates


# ============================================================
# Training & Prediction
# ============================================================

class LSTMTrainer:
    """Train and manage per-stock LSTM models."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(config.MODELS_DIR, exist_ok=True)

    def train(self, ticker, price_df, train_start, train_end,
              seq_len=None, horizon=None, epochs=100, batch_size=32, lr=0.001):
        """
        Train LSTM model for a single stock.

        Args:
            ticker: Stock ticker
            price_df: Full price DataFrame (with warmup data)
            train_start: Training period start "YYYY-MM-DD"
            train_end: Training period end "YYYY-MM-DD"
            seq_len: Input sequence length (default: config.LOOKBACK_SEQUENCE)
            horizon: Prediction horizon days (default: config.PREDICTION_HORIZON_DAYS)
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate

        Returns:
            dict with training results
        """
        seq_len = seq_len or config.LOOKBACK_SEQUENCE
        horizon = horizon or config.PREDICTION_HORIZON_DAYS

        print(f"  Training LSTM for {ticker}...")

        # Build features
        features_df = build_daily_features(price_df)
        train_mask = (features_df.index >= train_start) & (features_df.index <= train_end)
        train_features = features_df[train_mask]

        if len(train_features) < seq_len + horizon + 10:
            print(f"  {ticker}: Not enough training data ({len(train_features)} rows)")
            return None

        # Create sequences
        sequences, labels, dates = create_sequences(train_features, price_df, seq_len, horizon)
        if len(sequences) < 20:
            print(f"  {ticker}: Not enough sequences ({len(sequences)})")
            return None

        # Normalize features
        n_samples, n_steps, n_features = sequences.shape
        flat = sequences.reshape(-1, n_features)
        scaler = StandardScaler()
        flat_scaled = scaler.fit_transform(flat)
        sequences_scaled = flat_scaled.reshape(n_samples, n_steps, n_features)

        # Train/val split (80/20)
        split_idx = int(len(sequences_scaled) * 0.8)
        train_seqs = sequences_scaled[:split_idx]
        train_labels = labels[:split_idx]
        val_seqs = sequences_scaled[split_idx:]
        val_labels = labels[split_idx:]

        train_dataset = StockSequenceDataset(train_seqs, train_labels)
        val_dataset = StockSequenceDataset(val_seqs, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Model
        model = StockLSTM(input_size=n_features).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_acc = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    pred = model(X_batch)
                    predicted = (pred >= 0.5).float()
                    correct += (predicted == y_batch).sum().item()
                    total += len(y_batch)

            val_acc = correct / total * 100 if total > 0 else 0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}  loss={total_loss/len(train_loader):.4f}  val_acc={val_acc:.1f}%")

        # Save best model
        if best_state:
            model.load_state_dict(best_state)

        model_path = os.path.join(config.MODELS_DIR, f"{ticker}_lstm.pt")
        torch.save({
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "input_size": n_features,
            "seq_len": seq_len,
            "horizon": horizon,
        }, model_path)

        # Save metadata
        meta = {
            "ticker": ticker,
            "train_start": train_start,
            "train_end": train_end,
            "seq_len": seq_len,
            "horizon": horizon,
            "n_samples": len(sequences),
            "val_accuracy": round(best_val_acc, 1),
            "trained_at": datetime.now().isoformat(),
        }
        meta_path = os.path.join(config.MODELS_DIR, f"{ticker}_lstm_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  {ticker}: val_acc={best_val_acc:.1f}% ({len(sequences)} samples)")
        return meta

    def predict(self, ticker, date, price_df):
        """
        Predict using trained LSTM model.

        Args:
            ticker: Stock ticker
            date: Prediction date
            price_df: Price DataFrame

        Returns:
            P(up) probability 0-1, or 0.5 on failure
        """
        model_path = os.path.join(config.MODELS_DIR, f"{ticker}_lstm.pt")
        if not os.path.exists(model_path):
            return 0.5

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        seq_len = checkpoint["seq_len"]

        # Build features up to date
        if isinstance(date, str):
            date = pd.Timestamp(date)
        hist = price_df[price_df.index <= date]
        features_df = build_daily_features(hist)

        if len(features_df) < seq_len:
            return 0.5

        # Get last seq_len rows
        seq = features_df.iloc[-seq_len:].values

        # Scale using saved scaler parameters
        scaler_mean = np.array(checkpoint["scaler_mean"])
        scaler_scale = np.array(checkpoint["scaler_scale"])
        seq_scaled = (seq - scaler_mean) / scaler_scale

        # Predict
        model = StockLSTM(input_size=checkpoint["input_size"]).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        with torch.no_grad():
            x = torch.FloatTensor(seq_scaled).unsqueeze(0).to(self.device)
            prob_up = model(x).item()

        return prob_up

    def train_all(self, tickers, price_data_dict, train_start, train_end):
        """Train LSTM models for all tickers."""
        results = {}
        for ticker in tickers:
            if ticker not in price_data_dict:
                print(f"  {ticker}: No price data available")
                continue
            meta = self.train(ticker, price_data_dict[ticker], train_start, train_end)
            if meta:
                results[ticker] = meta
        return results
