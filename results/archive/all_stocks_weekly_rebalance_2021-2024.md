# All 10 Stocks Backtest - Weekly Rebalance (2021-2024)

## Test Environment
- Model: GPT-5.4 (OpenAI API)
- Period: 2021-01-01 to 2024-12-31
- Rebalance: Every 5 trading days (~weekly)
- Threshold: P(up) >= 52%
- Learning: All predictions (up + down) contribute to indicator learning
- Predictions per stock: ~189

## Results Summary

| Stock | Return | SPY | Alpha | Beat SPY? |
|-------|--------|-----|-------|-----------|
| NVDA | **+1209.1%** | +71.3% | **+1137.7%** | YES |
| TSLA | **+581.5%** | +71.3% | **+510.1%** | YES |
| AVGO | **+184.4%** | +71.3% | **+113.0%** | YES |
| LLY | **+154.9%** | +71.3% | **+83.6%** | YES |
| AAPL | **+130.8%** | +71.3% | **+86.0%** | YES |
| MSFT | **+119.1%** | +71.3% | **+47.8%** | YES |
| GOOGL | +54.3% | +71.3% | -17.1% | NO |
| JPM | +47.3% | +71.3% | -24.0% | NO |
| AMZN | +31.2% | +71.3% | -40.1% | NO |
| META | -76.5% | +71.3% | -147.8% | NO |

## Aggregate Statistics
- Stocks that beat SPY: **6 out of 10** (60%)
- Average return: **+243.6%**
- Average alpha: **+174.9%**
- SPY benchmark return: **+71.3%**
- Best performer: NVDA (+1209.1%)
- Worst performer: META (-76.5%)

## Key Observations
1. **High-momentum tech stocks dominated**: NVDA, TSLA, AVGO captured massive
   upside while avoiding major drawdowns through timely "down" predictions
2. **META was the outlier**: The system likely stayed bullish during META's 2022
   crash (-77% peak-to-trough), failing to predict the severity of the decline
3. **Financial (JPM) underperformed**: GPT-5.4 may be less effective at predicting
   bank stocks where macro factors (interest rates, yield curve) matter more than
   technical indicators
4. **6/10 positive alpha**: Majority of stocks generated meaningful alpha,
   suggesting the adaptive indicator learning system generalizes across sectors

## Date: 2026-04-03
