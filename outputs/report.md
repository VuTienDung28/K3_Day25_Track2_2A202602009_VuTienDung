# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## The GPU-Util Lie

`nvidia-smi` GPU-Util only reports that *some* kernel is resident during the sample
window — it measures clock activity, not FLOP throughput. An LLM decode phase is
memory-bound (arithmetic intensity ~1-2 FLOP/byte vs the H100 ridge point ~295):
the SMs sit stalled waiting on HBM, so the GPU reads ~98% busy while doing ~20% of
peak math. You pay for peak FLOPs and receive a fraction — the effective price per
useful FLOP is 1/MFU times the list price:

| GPU | Type | GPU-Util | MFU | Effective $/FLOP multiple |
|---|---|---|---|---|
| gpu-h100-4 | H100 | 98.2% | 0.194 | 5.2x |
| gpu-a10g-1 | A10G | 96.9% | 0.268 | 3.7x |

## Recommended Actions (by ROI)

| # | Action | Savings (USD/mo) | Effort | Why this order |
|---|---|---|---|---|
| 1 | Switch purchasing tiers (spot/reserved per 55% break-even) | $10,040 | low | 80% of total savings; contract change, no code |
| 2 | Right-size util-lie GPUs + kill idle | $1,255 | low | quick wins — downgrade or switch off within the week |
| 3 | Inference levers (cascade + prompt cache + batch) | $1,212 | medium | 83% unit-cost cut ($/1M-token); scales with traffic growth |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1

## Reasoning Budget

- Reasoning traffic: 8.4% of requests, 16.5% of tokens
- Cost share: 16.5% of optimized spend; energy share: 94.0% of Wh (~80x per query)

| Cap | Rerouted | $ saved | Wh saved |
|---|---|---|---|
| 10% | 0 | $0.00 | 0 |
| 5% | 81 | $0.73 | 15,778 |
| 2% | 153 | $0.96 | 25,556 |

- Routing rule: budget-cap reasoning traffic; when over cap, reroute the **largest thinkers first** — keep reasoning only where it pays for itself.

## Carbon-Aware Scheduling

- Interruptible workload energy: 1,789 kWh/month
- Current (us-east-1): 679,820 gCO2e/month
- Cleanest (europe-north1): 53,670 gCO2e/month -> save 626,150 gCO2e (92%)

| Region | $/kWh | gCO2/kWh | Electricity | Carbon |
|---|---|---|---|---|
| us-east-1 | $0.120 | 380 | $214.68 | 679,820 |
| us-west-2 | $0.070 | 120 | $125.23 | 214,680 |
| europe-north1 | $0.090 | 30 | $161.01 | 53,670 |
| europe-central2 | $0.180 | 660 | $322.02 | 1,180,740 |
| us-east-wa | $0.055 | 90 | $98.39 | 161,010 |

- Cheapest: us-east-wa · Cleanest: europe-north1 · Balanced: us-east-wa
- Carbon meets cost: moving to europe-north1 also saves $54/month of electricity — the carbon cut is (nearly) free.
- Latency trade-off: the cleanest region suits interruptible training/batch jobs;
  latency-sensitive inference should stay near users (e.g. us-east-wa / us-west-2).

_Figures are June-2026 as-of snapshots; re-baseline before acting._