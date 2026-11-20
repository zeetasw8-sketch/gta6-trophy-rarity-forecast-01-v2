"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its
earn rate on Exophase at the case's target, as five quantiles (5%, 25%,
50%, 75%, 95%), scored by pinball loss on the logit scale.

Model (deterministic, Python standard library only, ~0.1 s per case).
The observed quantities are counts: E_i(t) players who earned trophy i
and N(t) players Exophase tracks; the rate is their ratio. The median
forecast blends three estimators in logit space:

  (a) damped two-channel log-linear trends -- a weighted least-squares
      trend on log E_i and on log N over the last 48 h (12 h half-life),
      each slope shrunk toward its tier median and the global median
      with pseudo-counts (busy trophies trust their own slope, rare ones
      shrink hardest), extrapolated with exponential damping
      (tau = 1.5 days) so earners can saturate while tracking grows;
  (b) a rate-logistic fit -- rate(t) = C * sigmoid((t - t50) / 0.45 d),
      t50 on a grid, C in closed form in log space, ridge-averaged with
      temperature weights and log C shrunk toward the tier median;
  (c) an E-logistic fit -- the same S-curve fit on earner counts
      E_i(t), with N(t) from a shared logistic fit, so the forecast rate
      is E_target / N_target: this is the channel that sees dilution,
      when tracking growth outruns earning and rates fall at a refresh.

(b) is exact when the rate itself is an S-curve; (c) is exact when
adoption counts are S-curves and tracking growth dilutes the rate.
Weights: 62% (b), 33% (c), 5% (a). A trophy whose history carries no
usable signal falls back to the median day-7 forecast of its own tier.

Intervals (logit scale): random walk 0.70/sqrt(day) + slope standard
errors of both trend channels + fit residual + Poisson 1/sqrt(E) +
a refresh-jump term measured from N jumps in the history (0.6 when
none seen), times a 0.90 calibration. When the whole history is flat
(launch night, no refresh yet seen) the median is shifted down by
ln(6) -- the documented launch-night tracking jump 638 -> 3868 -- and
the interval widened x1.5: a refresh is coming and new players dilute
the rates. Medians are capped at tier ceilings (platinum 3%, gold 20%,
silver 60%, bronze 98%).

Usage: python3 main.py (reads $MAKEFUN_INPUT_DIR or /input, writes
$MAKEFUN_OUTPUT_DIR or /output, one <id>.out per <id>.in).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
T0 = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# --- tuned parameters (grid-searched on 7 final-like synthetic cases,
# 5 harder stress scenarios and a live-Exophase pseudo-case; see README).
WINDOW_H = 48.0        # WLS window, hours before the checkpoint
HALF_LIFE_H = 12.0     # exponential time-decay half-life, hours
TAU_E = 1.5            # damping timescale, earners, days
TAU_N = 1.5            # damping timescale, tracked players, days
K_TIER = 20.0          # pseudo-count pulling a slope to its tier median
K_GLOB = 5.0           # pseudo-count pulling a slope to the global median
SIGMA_RW = 0.70        # random-walk sigma per sqrt(day), logit units
SIGMA_CAL = 0.90       # calibration multiplier on total sigma
JUMP_SCALE = 1.0       # scale on the refresh-jump sigma term
JUMP_DEFAULT = 0.6     # jump size (logit) assumed when history shows none
FLAT_SHIFT = 1.8       # logit median shift down when history is flat
FLAT_MULT = 1.5        # sigma multiplier when history is flat
W_RLOG = 0.62          # weight of the rate-logistic channel
W_ELOG = 0.33          # weight of the E-logistic channel
LOG_SCALE = 0.45       # adoption-curve timescale, days
K_LOGC = 2.0           # shrinkage of fitted log C / log E_inf to tier median
LOG_MIN_GAIN = 0.10    # a logistic fit must beat a flat line by this much
LOG_S2 = 0.10          # SSE scale (log^2) for ridge weights
FALLBACK_SIG = 1.0     # extra sigma (logit) for no-signal trophies
TIER_CAP = {"platinum": 3.0, "gold": 20.0, "silver": 60.0, "bronze": 98.0}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg > 30:
        return 100.0
    if lg < -30:
        return 0.0
    return 100.0 / (1.0 + math.exp(-lg))


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def median(xs):
    s = sorted(xs)
    m = len(s)
    if m == 0:
        return 0.0
    return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])


def weighted_median(pairs):
    total = sum(w for _, w in pairs)
    if total <= 0:
        return median([v for v, _ in pairs])
    acc = 0.0
    for value, weight in sorted(pairs):
        acc += weight
        if acc >= 0.5 * total:
            return value
    return sorted(pairs)[-1][0]


def sigmoid(z):
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, se_b, resid_sd)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return (ys[-1] if ys else 0.0), 0.0, 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, 0.0
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(len(xs) - 2, 1)
    s2 = sum(w * r * r for w, r in zip(ws, resid)) / sw
    se_b = math.sqrt(s2 / sxx) if sxx > 0 else 0.0
    resid_sd = math.sqrt(sum(r * r for r in resid) / dof)
    return a, b, se_b, resid_sd


def rate_logistic_fit(points, tier):
    """Fit rate(t) = C * sigmoid((t - t50) / LOG_SCALE) to [(t, rate)].

    t50 runs over a 0..7 day grid; for each t50, log C is the mean of
    log(rate) - log(sigmoid(...)) over the points (closed form in log
    space, so tiny rates count as much as big ones). Returns the list
    of (sse, C, t50) candidates sorted by SSE, or None when the series
    is flat or the fit does not beat a flat line by LOG_MIN_GAIN.
    """
    rates = [r for _, r in points]
    if len(points) < 8 or max(rates) - min(rates) < 0.02:
        return None
    ys = [logit(r) for r in rates]
    med = median(ys)
    flat_sse = sum((y - med) ** 2 for y in ys)
    if flat_sse <= 1e-9:
        return None
    cands = []
    t50 = 0.0
    while t50 <= 7.0 + 1e-9:
        logs, use = [], []
        for t, r in points:
            sig = sigmoid((t - t50) / LOG_SCALE)
            if sig > 1e-6 and r > 0:
                logs.append(math.log(r) - math.log(sig))
                use.append((t, r, sig))
        if len(logs) >= 6:
            amp = math.exp(sum(logs) / len(logs))
            amp = min(amp, TIER_CAP[tier], 99.0)
            amp = max(amp, max(rates))
            sse = 0.0
            for t, r, sig in use:
                pp = amp / 100.0 * sig
                pred = math.log(pp / (1.0 - pp)) if 0 < pp < 1 else 9.0
                sse += (logit(r) - pred) ** 2
            cands.append((sse, amp, t50))
        t50 += 0.5
    if not cands:
        return None
    cands.sort()
    if cands[0][0] > (1.0 - LOG_MIN_GAIN) * flat_sse:
        return None
    return cands


def n_logistic_fit(times, counts):
    """Grid fit N(t) = N_inf / (1 + (N_inf/N0 - 1) exp(-k t)) on log N.

    Returns (N_inf, k) or None when N never moves.
    """
    if max(counts) - min(counts) <= 0:
        return None
    step = max(1, len(times) // 96)
    pts = list(zip(times, counts))[::step]
    n0 = counts[0]
    best = None
    for n_inf in (30000.0, 50000.0, 80000.0, 120000.0, 200000.0,
                  350000.0, 600000.0, 1000000.0):
        for j in range(19):
            k = 0.3 + 0.15 * j
            sse = 0.0
            for t, n in pts:
                pred = n_inf / (1.0 + (n_inf / n0 - 1.0) * math.exp(-k * t))
                sse += (math.log(max(n, 1.0)) - math.log(pred)) ** 2
            if best is None or sse < best[0]:
                best = (sse, n_inf, k)
    return best[1], best[2]


def e_logistic_fit(points, n_inf):
    """Fit E(t) = E_inf * sigmoid((t - t50) / LOG_SCALE) to [(t, E)].

    Same grid/closed-form scheme as rate_logistic_fit, in log counts.
    Returns [(sse, E_inf, t50)] sorted by SSE, or None.
    """
    es = [e for _, e in points]
    if len(points) < 8:
        return None
    logs_e = [math.log(e) for e in es]
    if max(logs_e) - min(logs_e) < 0.1:
        return None
    med = median(logs_e)
    flat_sse = sum((y - med) ** 2 for y in logs_e)
    if flat_sse <= 1e-9:
        return None
    cap = 0.99 * n_inf
    cands = []
    t50 = 0.0
    while t50 <= 7.0 + 1e-9:
        logs, use = [], []
        for t, e in points:
            sig = sigmoid((t - t50) / LOG_SCALE)
            if sig > 1e-6:
                logs.append(math.log(e) - math.log(sig))
                use.append((t, e, sig))
        if len(logs) >= 6:
            amp = math.exp(sum(logs) / len(logs))
            amp = min(amp, cap)
            amp = max(amp, max(es))
            sse = 0.0
            for t, e, sig in use:
                sse += (math.log(e) - math.log(max(amp * sig, 1e-9))) ** 2
            cands.append((sse, amp, t50))
        t50 += 0.5
    if not cands:
        return None
    cands.sort()
    if cands[0][0] > (1.0 - LOG_MIN_GAIN) * flat_sse:
        return None
    return cands


def ridge_predict(cands, t_target, shrink_log, cap):
    """Temperature-weighted median prediction over the (amp, t50) ridge.

    In the exponential phase many (amp, t50) pairs fit the history
    almost equally well but diverge at the target; averaging over them
    with weights exp(-(sse - sse_min) / (2 * LOG_S2)) is the honest
    point forecast. Each candidate's amplitude is first shrunk toward
    the tier median (log space). Also returns the ridge dispersion in
    log units (used only for diagnostics here).
    """
    sse0 = cands[0][0]
    preds = []
    for sse, amp, t50 in cands:
        a = amp
        if shrink_log is not None and K_LOGC > 0:
            a = math.exp((math.log(amp) + K_LOGC * shrink_log) / (1.0 + K_LOGC))
            a = min(a, cap)
        weight = math.exp(-(sse - sse0) / (2.0 * LOG_S2))
        preds.append((a * sigmoid((t_target - t50) / LOG_SCALE), weight))
    pred = weighted_median([(max(v, 1e-6), w) for v, w in preds])
    lgs = [math.log(max(v, 1e-6)) for v, _ in preds]
    mu_l = sum(lgs) / len(lgs)
    disp = math.sqrt(sum((x - mu_l) ** 2 for x in lgs) / len(lgs))
    return pred, disp


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    hist = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    t_end = parse_time(case["at"])
    t_cut = (t_end - T0).total_seconds() / 86400.0
    t_target = t_cut + horizon_days
    n_tr = len(case["trophies"])
    read_t = [(parse_time(h["at"]) - T0).total_seconds() / 86400.0 for h in hist]
    read_age_h = [(t_end - parse_time(h["at"])).total_seconds() / 3600.0 for h in hist]
    ns = [float(h["players_tracked"]) for h in hist]
    rates = [h["rates"] for h in hist]

    # Flat history: every rate and the tracked count never move.
    flat = True
    for i in range(n_tr):
        col = [rates[k][i] for k in range(len(hist)) if rates[k][i] is not None]
        if col and max(col) - min(col) > 0.05:
            flat = False
            break
    if flat and (max(ns) - min(ns) > 0):
        flat = False

    # --- tracked players: damped log-linear trend + logistic fit.
    xs, ys, ws = [], [], []
    for k in range(len(hist)):
        if read_age_h[k] > WINDOW_H:
            continue
        xs.append(-read_age_h[k] / 24.0)
        ys.append(math.log(max(ns[k], 1.0)))
        ws.append(0.5 ** (read_age_h[k] / HALF_LIFE_H))
    _, slope_n, se_n, _ = wls(xs, ys, ws)
    damp_n = TAU_N * (1.0 - math.exp(-horizon_days / TAU_N))
    log_n_target = math.log(max(ns[-1], 1.0)) + slope_n * damp_n
    n_fit = n_logistic_fit(read_t, ns)
    if n_fit is not None:
        n_inf, n_k = n_fit
        n0 = ns[0]
        n_target = n_inf / (1.0 + (n_inf / n0 - 1.0) * math.exp(-n_k * t_target))
    else:
        n_inf, n_target = None, math.exp(log_n_target)

    # --- earners: damped log-linear trend per trophy, slopes pooled.
    raw_slopes, fits = [], []
    for i in range(n_tr):
        xs, ys, ws = [], [], []
        for k in range(len(hist)):
            r = rates[k][i]
            if r is None or read_age_h[k] > WINDOW_H:
                continue
            e = max(r / 100.0 * ns[k], 0.5)
            xs.append(-read_age_h[k] / 24.0)
            ys.append(math.log(e))
            ws.append(0.5 ** (read_age_h[k] / HALF_LIFE_H) * (e / (e + 10.0)))
        a, b, se_b, resid = wls(xs, ys, ws)
        raw_slopes.append(b)
        last_rate = rates[-1][i]
        e_last = max((last_rate if last_rate is not None else 0.0) / 100.0 * ns[-1], 0.5)
        fits.append((a, b, se_b, resid, e_last))
    glob_med = median(raw_slopes)
    tier_med = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        ss = [raw_slopes[i] for i, tr in enumerate(case["trophies"]) if tr["tier"] == tier]
        tier_med[tier] = median(ss) if ss else glob_med

    # --- refresh jumps: typical |d logit rate| at reads where N jumps >3%.
    jumps = []
    for k in range(1, len(hist)):
        if ns[k - 1] > 0 and ns[k] / ns[k - 1] > 1.03:
            ds = []
            for i in range(n_tr):
                r0, r1 = rates[k - 1][i], rates[k][i]
                if r0 is not None and r1 is not None:
                    ds.append(abs(logit(r1) - logit(r0)))
            if ds:
                jumps.append(median(ds))
    jump_size = median(jumps) if jumps else JUMP_DEFAULT
    jump_term = JUMP_SCALE * jump_size * math.sqrt(min(horizon_days, 3.0))
    damp_e = TAU_E * (1.0 - math.exp(-horizon_days / TAU_E))

    # --- logistic fits, both channels (subsampled to <= 48 reads).
    step = max(1, len(hist) // 48)
    r_fits, e_fits = [None] * n_tr, [None] * n_tr
    for i, tr in enumerate(case["trophies"]):
        if W_RLOG > 0:
            pts = [(read_t[k], rates[k][i]) for k in range(0, len(hist), step)
                   if rates[k][i] is not None and rates[k][i] > 0]
            r_fits[i] = rate_logistic_fit(pts, tr["tier"])
        if W_ELOG > 0 and n_inf is not None:
            pts = [(read_t[k], max(rates[k][i] / 100.0 * ns[k], 0.5))
                   for k in range(0, len(hist), step)
                   if rates[k][i] is not None and rates[k][i] > 0]
            e_fits[i] = e_logistic_fit(pts, n_inf)
    tier_log_c, tier_log_e = {}, {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = [math.log(f[0][1]) for i, f in enumerate(r_fits)
                if f is not None and case["trophies"][i]["tier"] == tier]
        tier_log_c[tier] = median(vals) if vals else None
        vals = [math.log(f[0][1]) for i, f in enumerate(e_fits)
                if f is not None and case["trophies"][i]["tier"] == tier]
        tier_log_e[tier] = median(vals) if vals else None

    # --- first pass: each trophy's own-signal median (logit).
    w_two = max(0.0, 1.0 - W_RLOG - W_ELOG)
    own_mu, own_fit, own_extra = [], [], []
    for i, tr in enumerate(case["trophies"]):
        a, b, se_b, resid, e_last = fits[i]
        b_pool = (b * e_last + tier_med[tr["tier"]] * K_TIER
                  + glob_med * K_GLOB) / (e_last + K_TIER + K_GLOB)
        log_e_target = math.log(e_last) + b_pool * damp_e
        rate_two = 100.0 * math.exp(log_e_target) / math.exp(log_n_target)
        mu = w_two * logit(rate_two)
        wsum = w_two
        has_fit = False
        if W_RLOG > 0 and r_fits[i] is not None:
            pred, _ = ridge_predict(r_fits[i], t_target, tier_log_c[tr["tier"]],
                                    TIER_CAP[tr["tier"]])
            mu += W_RLOG * logit(pred)
            wsum += W_RLOG
            has_fit = True
        if W_ELOG > 0 and e_fits[i] is not None:
            pred, _ = ridge_predict(e_fits[i], t_target, tier_log_e[tr["tier"]],
                                    0.99 * n_inf)
            rate_elog = 100.0 * pred / max(n_target, 1.0)
            mu += W_ELOG * logit(rate_elog)
            wsum += W_ELOG
            has_fit = True
        if wsum == 0:
            mu, wsum = logit(rate_two), 1.0
        mu /= wsum
        own_mu.append(mu)
        own_fit.append(has_fit)
        own_extra.append((a, b, se_b, resid, e_last))
    # Tier prior for no-signal trophies: the median day-7 forecast of the
    # same tier's trophies that did carry a signal. A trophy flat at
    # ~0.03% on day 2.7 is usually a late bloomer, not a dead trophy.
    tier_prior = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = [own_mu[i] for i, tr in enumerate(case["trophies"])
                if tr["tier"] == tier and own_fit[i]]
        tier_prior[tier] = median(vals) if vals else None

    rows = []
    for i, tr in enumerate(case["trophies"]):
        a, b, se_b, resid, e_last = own_extra[i]
        if own_fit[i]:
            mu = own_mu[i]
            extra_sig2 = 0.0
        else:
            mu = tier_prior[tr["tier"]] if tier_prior[tr["tier"]] is not None else own_mu[i]
            extra_sig2 = FALLBACK_SIG ** 2
        rate_med = inv_logit(mu)
        rate_med = min(rate_med, TIER_CAP[tr["tier"]])
        mu = logit(rate_med)
        if flat:
            mu -= FLAT_SHIFT
        sig2 = ((SIGMA_RW * math.sqrt(horizon_days)) ** 2
                + (se_b * damp_e) ** 2 + (se_n * damp_n) ** 2
                + max(resid, 0.05) ** 2
                + 1.0 / max(e_last, 1.0)
                + jump_term ** 2
                + extra_sig2)
        sig = math.sqrt(sig2) * SIGMA_CAL
        if flat:
            sig *= FLAT_MULT
        sig = max(sig, 0.05)
        q = [round(clamp_pct(inv_logit(mu + z * sig)), 4) for z in Z]
        for k in range(1, 5):
            q[k] = max(q[k], q[k - 1])
        rows.append({"id": tr["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
