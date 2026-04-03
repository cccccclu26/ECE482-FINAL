# AAPL Out-of-Sample Backtest (2018-2021)

## Purpose
Validate that indicator weights learned from 2021-2024 training period
generalize to unseen 2018-2021 data (out-of-sample test).

## Test Environment
- Model: GPT-5.4 (OpenAI API)
- Period: 2018-01-02 to 2020-12-31 (out-of-sample)
- Training Period: 2021-04-05 to 2024-12-30 (in-sample, learned prior)
- Rebalance: Every 5 trading days (~weekly)
- Threshold: P(up) >= 52%
- Starting indicator weights: Carried over from 2021-2024 training
- Total predictions: 152

## Results
| Metric | LLM Portfolio | SPY Benchmark |
|--------|--------------|---------------|
| Final Value | $41,279 | $14,713 |
| Total Return | **+312.8%** | +47.1% |
| CAGR | **+60.0%** | +13.7% |
| Alpha | **+265.7%** | — |

## Key Observations
1. **COVID Crash (Feb-Mar 2020):** GPT-5.4 predicted "down" for 4 consecutive
   weeks (Feb 19 - Mar 18), holding cash through -7.93%, -12.31%, -12.52% SPY drops
2. **Post-COVID Recovery:** Correctly switched to "up" and captured massive gains
   (+14.9% in one week during Aug 2020 Apple stock split run-up)
3. **2018 Q4 Selloff:** Held cash through Oct-Dec 2018 downturn, avoiding ~15% drawdown
4. **Learned weights transferred well:** Indicator weights from 2021-2024 proved
   predictive on completely unseen 2018-2021 data

## Comparison: In-Sample vs Out-of-Sample
| Metric | In-Sample (2021-2024) | Out-of-Sample (2018-2021) |
|--------|----------------------|--------------------------|
| Total Return | +130.8% | +312.8% |
| CAGR | +25.0% | +60.0% |
| Alpha | +86.0% | +265.7% |
| Predictions | 189 | 152 |

## Date: 2026-04-03
