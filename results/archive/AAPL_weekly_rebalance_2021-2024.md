# AAPL Backtest - Weekly Rebalance (5 trading days)

## Test Environment
- Model: GPT-5.4 (OpenAI API)
- Period: 2021-04-05 to 2024-12-30
- Rebalance: Every 5 trading days (~weekly)
- Threshold: P(up) >= 52%
- Learning: All predictions (up + down) contribute to indicator learning
- Total predictions: 189 (indicator_history capped at 50 most recent)

## Results
| Metric | LLM Portfolio | SPY Benchmark |
|--------|--------------|---------------|
| Final Value | $23,077 | $14,475 |
| Total Return | +130.8% | +44.8% |
| CAGR | +25.0% | +10.4% |
| Alpha | +86.0% | — |

## Learned Indicator Weights (from last 50 predictions)
| Indicator | Weight |
|-----------|--------|
| ema_alignment | 0.611 |
| macd | 0.609 |
| sma_trend | 0.605 |
| adx | 0.602 |
| williams_r | 0.587 |
| news_sentiment | 0.581 |
| stochastic | 0.581 |
| golden_cross | 0.575 |
| volume_ratio | 0.575 |
| bollinger_bands | 0.568 |
| obv | 0.560 |
| rsi | 0.559 |
| cci | 0.552 |
| atr | 0.439 |

## Learned Sentiment Ratio
- Sentiment: 60.2%
- Technical: 39.8%

## Comparison vs Monthly Rebalance
| Metric | Weekly | Monthly |
|--------|--------|---------|
| Total Return | +130.8% | +98.2% |
| CAGR | +25.0% | +20.0% |
| Alpha | +86.0% | +53.4% |
| Predictions | 189 | 45 |

## Date: 2026-04-03
