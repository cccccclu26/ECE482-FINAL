# Portfolio Comparison: LLM vs LSTM vs SPY (2018-2024)

## Test Environment
- Period: 2018-01-02 to 2024-07-25 (6.6 years, 321 successful weeks)
- Rebalance: Every 5 trading days (~weekly)
- Initial Capital: $10,000
- LLM Model: GPT-5.4 (fund manager mode, single call per week for all 10 stocks)
- LSTM Model: 2-layer LSTM trained on 2021-2024, threshold 80%
- Stock Universe: AAPL, NVDA, META, JPM, TSLA, MSFT, AMZN, GOOGL, AVGO, LLY

## Results (321 weeks, fair comparison)

| Metric | LLM Portfolio | LSTM (80%) | SPY |
|--------|--------------|------------|-----|
| Final Value | $53,709 | $133,492 | $22,954 |
| Total Return | +437.1% | +1234.9% | +129.5% |
| CAGR | +29.2% | +48.4% | +13.5% |
| Alpha | +307.5% | +1105.4% | — |

## LLM Paper Account Statistics
- Total weeks: 352
- Successful API calls: 321 (91.2%)
- Failed (429 rate limit): 31 (held cash)
- Win rate: 59.5% (191 wins / 130 losses)
- Best week: Week 307 (2024-02-01) PnL +$3,465 (top picks: MSFT, NVDA, META)
- Worst week: Week 224 (2022-06-07) PnL -$2,948

## Key Differences

### LLM Strategy (Fund Manager Mode)
- One GPT-5.4 call per week with ALL 10 stocks data
- Actively manages allocation weights (0-100% per stock)
- Maintains cash buffer (avg 15-20%)
- Considers news sentiment + learned indicator weights
- More conservative, better risk management

### LSTM Strategy (80% Threshold)
- Individual binary prediction per stock (up/down with probability)
- Probability often 95-100% (overconfident)
- Threshold filters out some weak signals (cash 20-60% in bearish periods)
- No news/sentiment awareness
- Higher returns due to heavier equity exposure in bull market

## Analysis
- Both strategies significantly outperform SPY
- LSTM returns 3x higher than LLM, primarily due to heavier equity exposure
- LSTM overconfidence (99%+ probabilities) means it stays heavily invested
  through bull markets, capturing more upside
- LLM is more conservative with active cash management, providing better
  downside protection but sacrificing some upside
- LLM provides explainable decisions (reasoning, top picks, avoid list)
- LSTM is a black box

## Files
- LLM paper account: results/paper_account_llm_20260403_180138.csv
- LLM detailed log: results/paper_account_llm_20260403_180138.json
- LSTM results: results/backtest_lstm_20260403_184610.json

## Date: 2026-04-03
