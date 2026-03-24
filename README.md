# LLM-Driven Stock Prediction System

**ECE 482 Senior Design Project** — University of Miami, Spring 2026

An exploratory stock prediction system that uses LLMs (Claude 3.7 Sonnet + GPT-5) as the **core decision maker**, not just a sentiment scorer. The system feeds technical indicators and news directly to the LLM, lets it output structured predictions, and iteratively refines prompts based on backtest results. An LSTM neural network serves as a data-driven baseline for comparison.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Code (Memory Layer)                    │
│                                                         │
│  memory/stock_context/   ← per-stock learned patterns   │
│  memory/prompt_history/  ← prompt version tracking      │
│  memory/eval_logs/       ← prediction accuracy logs     │
│                                                         │
│  Each LLM call gets everything it needs in the prompt   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              LLM Prediction (Stateless)                  │
│                                                         │
│  Prompt = Technical Data + News + Stock Context          │
│                                                         │
│  Claude 3.7 Sonnet ──┐                                  │
│                      ├── Ensemble Average → Prediction   │
│  GPT-5 ─────────────┘                                  │
│                                                         │
│  Output: {direction, probability, target_return,         │
│           key_factors, risk_factors, reasoning}          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Iterative Refinement Loop                    │
│                                                         │
│  1. Run predictions with prompt vN                       │
│  2. Backtest → evaluate accuracy                         │
│  3. Analyze errors (which stocks, which conditions)      │
│  4. Update prompt template → vN+1                        │
│  5. Update per-stock context (learned biases)            │
│  6. Repeat                                              │
└─────────────────────────────────────────────────────────┘
```

### LLM vs LSTM Comparison

| Aspect | LLM Predictor | LSTM Baseline |
|--------|--------------|---------------|
| Input | Technical + News + Context | Price sequences only |
| Model | Claude 3.7 + GPT-5 ensemble | PyTorch LSTM (2-layer) |
| Training | Prompt engineering (iterative) | Gradient descent (backprop) |
| Explainability | Full reasoning in natural language | Black box |
| Innovation | Prompt refinement methodology | Standard deep learning |

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
POLYGON_API_KEY=your_polygon_key
WAVESPEED_API_KEY=your_wavespeed_key
```

### 3. Run

```bash
# Predict single stock (today)
python main.py predict --ticker AAPL

# Predict all 10 stocks
python main.py predict --all

# Backtest LLM portfolio vs SPY
python main.py backtest --method llm --start 2025-01-01 --end 2026-03-01

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

## Stock Universe

10 stocks spanning tech, finance, and healthcare:

AAPL, NVDA, META, JPM, TSLA, MSFT, AMZN, GOOGL, AVGO, LLY

---

## Key Design Decisions

- **LLM as decision maker, not tool**: The LLM receives all data and outputs the prediction directly. This makes prompt engineering the core research contribution.
- **Code is the memory**: LLMs are stateless via API. All learned patterns, biases, and history are stored in JSON files and injected into each prompt.
- **Prompt versioning**: Each prompt template is versioned (v1, v2, ...). Backtests track which version was used, enabling systematic comparison.
- **Dual-model ensemble**: Claude and GPT-5 analyze the same data independently. Averaging reduces single-model bias; disagreement signals uncertainty.
- **LSTM as baseline**: A standard neural network trained on price sequences provides a data-driven comparison point for the LLM approach.
- **Per-stock context**: Each stock accumulates learned patterns (e.g., "JPM shows contrarian sentiment behavior") that are injected into future prompts.

---

## Project Structure

```
ECE482-FINAL/
├── main.py              # CLI entry point (predict/backtest/train/compare)
├── config.py            # API keys, tickers, parameters
├── llm_predictor.py     # Core: builds prompts, calls LLM, parses predictions
├── llm_engine.py        # WaveSpeed API wrapper, ensemble logic
├── memory.py            # JSON-based memory (stock context, prompt history, eval logs)
├── technical.py         # EMA25/50/100 + RSI(14) + volume ratio
├── data_fetcher.py      # Polygon.io + yfinance price/news fetcher
├── lstm_model.py        # PyTorch LSTM model (train + predict)
├── backtest.py          # Backtest framework (LLM vs LSTM vs SPY)
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not committed)
├── prompts/
│   └── v1.txt           # Prompt template version 1
├── memory/
│   ├── stock_context/   # Per-stock learned patterns (JSON)
│   ├── prompt_history/  # Prompt version performance tracking
│   └── eval_logs/       # Individual prediction logs
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
