# LLM-Driven Stock Prediction System

**ECE 482 Senior Design Project** — University of Miami, Spring 2026

An exploratory stock prediction system that uses GPT-5.4 as the **core decision maker**. The system feeds all mainstream technical indicators and news directly to the LLM, lets it output structured predictions with indicator importance ratings, and iteratively learns which indicators work best for each stock. An LSTM neural network serves as a data-driven baseline for comparison.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Adaptive Memory Layer                      │
│                                                             │
│  memory/stock_context/   ← per-stock indicator weights +    │
│                            sentiment ratio (learned)        │
│  memory/prompt_history/  ← prompt version tracking          │
│  memory/eval_logs/       ← prediction logs with             │
│                            indicator_importance data         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Pipeline                                   │
│                                                             │
│  Polygon.io + yfinance → OHLCV price data                   │
│  Polygon.io → News articles (7-day lookback)                │
│                                                             │
│  Technical Indicators (14 types):                           │
│    Trend:      EMA(25/50/100), SMA(20/50/200), MACD, ADX   │
│    Momentum:   RSI(14), Stochastic, Williams %R, CCI        │
│    Volatility: Bollinger Bands, ATR                         │
│    Volume:     Volume Ratio, OBV                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM Prediction (GPT-5.4)                        │
│                                                             │
│  Prompt = All Indicators + News + Learned Stock Context     │
│                                                             │
│  Output:                                                    │
│    direction, probability, target_return,                    │
│    key_factors, risk_factors, reasoning,                     │
│    indicator_importance: {macd: 0.8, rsi: 0.6, ...},  ◀ NEW│
│    sentiment_weight: 0.3                               ◀ NEW│
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Adaptive Learning Loop                          │
│                                                             │
│  1. Run predictions with all indicators                      │
│  2. Backtest → evaluate accuracy                             │
│  3. Update per-stock indicator weights:                      │
│     - Correct prediction → cited indicators weight ↑         │
│     - Wrong prediction → cited indicators weight ↓           │
│  4. Update sentiment vs technical ratio per stock            │
│  5. Next prediction sees learned weights in context          │
│  6. LLM adapts which indicators to emphasize                 │
└─────────────────────────────────────────────────────────────┘
```

### LLM vs LSTM Comparison

| Aspect | LLM Predictor | LSTM Baseline |
|--------|--------------|---------------|
| Input | All technical indicators + News + Learned context | Price sequences only (7 features) |
| Model | GPT-5.4 (OpenAI API) | PyTorch LSTM (2-layer, hidden=64) |
| Training | Adaptive indicator learning (iterative) | Gradient descent (backprop) |
| Explainability | Full reasoning + indicator importance scores | Black box |
| Innovation | Learns optimal indicators per stock | Standard deep learning |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/cccccclu26/ECE482-FINAL.git
cd ECE482-FINAL
pip install -r requirements.txt
```

### 2. Set up API keys

Create a `.env` file:
```
OPENAI_API_KEY=your_openai_key
POLYGON_API_KEY=your_polygon_key
```

### 3. Run

```bash
# Predict single stock (today)
python main.py predict --ticker AAPL

# Predict all 10 stocks
python main.py predict --all

# Backtest LLM portfolio vs SPY
python main.py backtest --method llm --start 2021-01-01 --end 2024-12-31

# Backtest single stock
python main.py backtest --method llm --start 2021-01-01 --end 2024-12-31 --tickers AAPL

# Train LSTM models
python main.py train-lstm --start 2021-01-01 --end 2024-12-31

# Backtest LSTM portfolio vs SPY
python main.py backtest --method lstm --start 2025-01-01 --end 2026-03-01

# Compare LLM vs LSTM vs SPY
python main.py compare --start 2025-01-01 --end 2026-03-01

# Show prediction accuracy
python main.py accuracy

# System info
python main.py info
```

---

## Backtest Results

### AAPL (2021-04 to 2024-12, monthly rebalance)

| Metric | LLM Portfolio | SPY Benchmark |
|--------|--------------|---------------|
| Final Value | $16,785 | $14,475 |
| Total Return | **+67.9%** | +44.8% |
| CAGR | **+14.8%** | +10.4% |
| Alpha | **+23.1%** | — |

Key observations:
- GPT-5.4 correctly called bearish periods in 2022, holding cash to avoid major drawdowns
- Strong upside capture during recovery periods (+17.4% in Jul 2022, +16.3% in Jun 2024)
- Adaptive indicator learning improved predictions over the backtest period

---

## Stock Universe

10 stocks spanning tech, finance, and healthcare:

AAPL, NVDA, META, JPM, TSLA, MSFT, AMZN, GOOGL, AVGO, LLY

---

## Key Design Decisions

- **LLM as decision maker, not tool**: GPT-5.4 receives all data and outputs the prediction directly. Prompt engineering is the core research contribution.
- **All indicators, then learn**: Instead of pre-selecting indicators, we give the LLM ALL mainstream indicators and let it learn which ones matter per stock through iterative backtesting.
- **Adaptive indicator weights**: After each prediction is evaluated, the system updates per-stock indicator importance scores. Indicators cited in correct predictions gain weight; those in wrong predictions lose weight.
- **Sentiment ratio learning**: The system tracks the optimal balance between technical analysis and news sentiment for each stock, adapting over time.
- **Code is the memory**: LLMs are stateless via API. All learned patterns — indicator weights, sentiment ratios, prediction history — are stored in JSON files and injected into each prompt.
- **Prompt versioning**: Each prompt template is versioned (v1, v2, ...). Backtests track which version was used, enabling systematic comparison.
- **LSTM as baseline**: A standard neural network trained on price sequences provides a data-driven comparison point for the LLM approach.

---

## Technical Indicators

The system computes 14 types of indicators, grouped by category:

| Category | Indicators |
|----------|-----------|
| **Trend** | EMA(25/50/100), SMA(20/50/200), MACD(12,26,9), ADX(14), Golden Cross |
| **Momentum** | RSI(14), Stochastic(14,3,3), Williams %R(14), CCI(20) |
| **Volatility** | Bollinger Bands(20,2), ATR(14) |
| **Volume** | Volume Ratio (20-day), OBV trend |

All indicators are computed with look-ahead bias prevention (`as_of_date` cutoff).

---

## Project Structure

```
ECE482-FINAL/
├── main.py              # CLI entry point (predict/backtest/train/compare)
├── config.py            # API keys, model config, parameters
├── llm_predictor.py     # Core: builds prompts, calls LLM, parses predictions
├── llm_engine.py        # OpenAI GPT-5.4 API wrapper (via requests)
├── memory.py            # Adaptive memory (indicator weights, sentiment ratio, eval logs)
├── technical.py         # All 14 technical indicators (trend/momentum/volatility/volume)
├── data_fetcher.py      # Polygon.io + yfinance price/news fetcher
├── lstm_model.py        # PyTorch LSTM model (train + predict)
├── backtest.py          # Backtest framework (LLM vs LSTM vs SPY)
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not committed)
├── prompts/
│   └── v1.txt           # Prompt template with indicator importance + sentiment weight
├── memory/
│   ├── stock_context/   # Per-stock learned indicator weights + sentiment ratio
│   ├── prompt_history/  # Prompt version performance tracking
│   └── eval_logs/       # Prediction logs with indicator_importance data
├── models/              # Trained LSTM models (.pt files)
└── results/             # Backtest output (JSON)
```

---

## Team

- **Zonglu Chen** — LLM Pipeline, Prompt Engineering & Backtesting
- **Jorge Garzon** — Technical Analysis & Data Pipeline
- **Alexander Pena** — System Integration & Documentation
- **Advisor**: Dr. Mingzhe Chen

*ECE 481/482 Senior Design — University of Miami, 2025-2026*
