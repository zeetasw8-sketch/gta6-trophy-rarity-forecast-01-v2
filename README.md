# GTA VI trophy rarity forecaster — v2 (make.fun gta6-trophy-rarity-forecast-01)

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) [![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zeetasw8-sketch/gta6-trophy-rarity-forecast-01-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) [![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zeetasw8-sketch/gta6-trophy-rarity-forecast-01-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PS5 trophy's earn rate on Exophase at the case's
target, as five quantiles per trophy (5%, 25%, 50%, 75%, 95%), scored by
pinball loss on the logit scale. Python 3.13 standard library only,
deterministic, ~0.1 s per case.

## The model

The observed quantities are **counts**: E_i(t) players who earned trophy i
and N(t) players Exophase tracks; the rate is their ratio. The median
forecast blends three estimators in logit space (weights 62% / 33% / 5%):

- **(a) Damped two-channel log-linear trends.** Weighted least squares on
  log E_i and on log N over the last 48 h (12 h half-life). Each slope is
  shrunk toward its tier median and the global median with pseudo-counts
  (K_tier = 20, K_glob = 5): busy trophies trust their own slope, rare
  ones shrink hardest. Extrapolation is damped (tau = 1.5 days), so
  earners can saturate while tracking keeps growing.
- **(b) Rate-logistic fit.** rate(t) = C · sigmoid((t − t50) / 0.45 d),
  t50 on a 0–7 day grid, C in closed form in log space, ridge-averaged
  with temperature weights, log C shrunk toward the tier median.
  Exact when the rate itself is an S-curve.
- **(c) E-logistic fit.** The same S-curve fit on earner counts E_i(t),
  with N(t) from a shared logistic fit, so the forecast rate is
  E_target / N_target. This is the channel that sees **dilution**: when
  tracking growth outruns earning, rates fall at a refresh. Exact when
  adoption counts are S-curves.

(b) and (c) hedge two different worlds; the blend beats either alone on
every scenario family tested. A trophy whose history carries no usable
signal falls back to the median day-7 forecast of its own tier — a
trophy flat at ~0.03% on day 2.7 is usually a late bloomer, not a dead
trophy.

**Intervals** (logit scale): random walk 0.70/√day + slope standard
errors of both trend channels + fit residual + Poisson 1/√E + a
refresh-jump term measured from N jumps in the history (0.6 when none
seen), times a 0.90 calibration. When the whole history is flat (launch
night, no refresh yet seen) the median is shifted down by ln(6) — the
documented launch-night tracking jump 638 → 3868 — and the interval
widened ×1.5: a refresh is coming and new players dilute the rates.
Medians are capped at tier ceilings (platinum 3%, gold 20%, silver 60%,
bronze 98%).

## Backtest results (honest, same cases for every model)

Pinball loss on the logit scale (lower is better). v1 numbers are the
**reported** v1 results; v1-prose is my re-implementation of v1 from its
prose description only (never opened the v1 entry) — it does not
reproduce v1 exactly (pseudo 1.69 vs reported 1.89; synthetic mean 1.28
vs reported 1.04), so the reported numbers stay the benchmark.

| Case | Starter | v1-prose | v2 (this entry) | Reported v1 |
| --- | --- | --- | --- | --- |
| Pseudo p01 → live Exophase 19 Nov 18:49Z (919 min) | 2.337 | 1.687 | **0.661** | 1.891 |
| Pseudo p01 → live Exophase 20 Nov 03:51Z (1461 min) | 2.266 | 1.492 | **0.681** | — |
| Synthetic final-like, 7 scenarios, mean | 1.670 | 1.280 | **0.601** | 1.04 |
| Synthetic final-like, 7 scenarios, worst | 3.53 | 2.54 | **0.81** | 2.04 |
| New harder scenarios, 5 (spike / quiet+noisy / plateau / late-bloom / refresh-storm), mean | 2.259 | 1.867 | **0.841** | — |
| All 14 backtest cases, mean | 1.971 | 1.534 | **0.696** | — |

v2 wins 12 of the 14 cases head-to-head against v1-prose; the two losses
are persistence-friendly cases (mild-early 0.48 vs 0.42, plateau 0.64 vs
0.45) where every trend model overshoots and the starter wins too.
Mean 90% coverage across the 14 cases is 0.95. The live snapshot was
fetched 20 Nov 03:51Z via a text reader (direct fetch: Cloudflare 403);
its 55 rates were identical to the 19 Nov 18:49Z snapshot — the page
had not refreshed between the two fetches.

**Verdict: v2 is clearly stronger than v1** — ~65% lower loss on the
pseudo-case, ~42% lower on the synthetic final-like mean, ~60% lower
worst case, and better on the five harder stress scenarios too.

## How to run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes one `<id>.out` per
case. Two runs on the same case write identical bytes (verified on the
practice cases p00/p01: byte-identical, 0.07 s per run, well inside the
8 s cap). Output is at most 65,536 bytes per case.

## Files

- `main.py` — the whole model (single file, stdlib only)
- `LICENSE` — MIT
- `.makefun/owner.json` — payout wallet
