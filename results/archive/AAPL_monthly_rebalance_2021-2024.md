# AAPL Backtest - Monthly Rebalance (21 trading days)

## Test Environment
- Model: GPT-5.4 (OpenAI API)
- Period: 2021-04-05 to 2024-12-30
- Rebalance: Every 21 trading days (~monthly)
- Threshold: P(up) >= 52%
- Learning: All predictions (up + down) contribute to indicator learning
- Total predictions: 45

## Results
| Metric | LLM Portfolio | SPY Benchmark |
|--------|--------------|---------------|
| Final Value | $19,815 | $14,475 |
| Total Return | +98.2% | +44.8% |
| CAGR | +20.0% | +10.4% |
| Alpha | +53.4% | — |

## Learned Indicator Weights (after 45 predictions)
| Indicator | Weight |
|-----------|--------|
| news_sentiment | 0.619 |
| rsi | 0.614 |
| cci | 0.610 |
| atr | 0.607 |
| stochastic | 0.602 |
| macd | 0.589 |
| williams_r | 0.577 |
| adx | 0.576 |
| bollinger_bands | 0.576 |
| ema_alignment | 0.575 |

## Learned Sentiment Ratio
- Sentiment: 69.8%
- Technical: 30.2%

## Date: 2026-04-03
