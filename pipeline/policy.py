"""
policy.py - what pump prices and inflation would have been without the interventions.

For every episode with a non-tax measure (price cap, maximum price, margin cap, discount,
Hungary's post-cap window), per country and fuel:

  1. Market price without the measure: the Stage 1 normal-times model (long-run relation
     and weekly dynamics, estimated on market weeks only) is started from the actual
     pre-tax price in the week before the episode and fed the actual Brent path, week by
     week, until `after_weeks` weeks after the episode ends.
  2. Two counterfactual pump prices:
       market  counterfactual pre-tax price + the taxes actually charged
               -> isolates the cap / margin cap / discount
       full    counterfactual pre-tax price + VAT and per-litre taxes as they were
               `tax_baseline_weeks_before` weeks before the episode (tax cuts often started a few
               days before the price measures) -> adds the effect of tax changes made around it
  3. Monthly: the HICP fuels index tracks pump prices almost one-for-one (Stage 2 chain),
     so the gap becomes counterfactual fuel inflation and, with the fuel weight, a direct
     effect on headline inflation (pp). Second-round effects are not included.

Writes outputs/results/policy_counterfactual.csv (long format, `level` = weekly / monthly).
"""
import numpy as np
import pandas as pd

from pipeline import utils
from pipeline.stage1 import ecm_frame, fit_ecm, long_run, masks, sample_slice


# --- Episodes -----------------------------------------------------------------------------

def episodes(g, types):
    """Contiguous blocks of weeks with any non-tax measure in force (positions in g)."""
    on = g[[f"policy_{t}" for t in types]].any(axis=1).values
    blocks, start = [], None
    for i, flag in enumerate(on):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            blocks.append((start, i - 1))
            start = None
    if start is not None:
        blocks.append((start, len(on) - 1))
    out = []
    for a, b in blocks:
        names = sorted({n for v in g["intervention"].iloc[a:b + 1] for n in v.split(";") if n})
        out.append((a, b, "+".join(names)))
    return out


# --- Counterfactual simulation ---------------------------------------------------------------------

def simulate(g, s, lr, sr, p, q, start, stop):
    """Pre-tax price path from week `start` to `stop` under normal-times dynamics, actual Brent."""
    y = g[s["y"]].to_numpy(dtype=float)
    x = g[s["x"]].to_numpy(dtype=float)
    trend = np.arange(len(g)) / 52.0
    target = lr.params["const"] + lr.params["trend"] * trend + lr.params["x"] * x
    dx = np.r_[np.nan, np.diff(x)]
    cf = y.copy()
    dcf = np.r_[np.nan, np.diff(y)]
    P = sr.params
    for t in range(start, stop + 1):
        d = P["const"] + P["gap1"] * (cf[t - 1] - target[t - 1]) + P["dx0"] * dx[t]
        d += sum(P[f"dy{i}"] * dcf[t - i] for i in range(1, p + 1))
        d += sum(P[f"dx{j}"] * dx[t - j] for j in range(1, q + 1))
        cf[t] = cf[t - 1] + d
        dcf[t] = d
    out = np.full(len(g), np.nan)
    out[start:stop + 1] = cf[start:stop + 1]
    return out


def weekly_counterfactuals(cfg, weekly, s1):
    s, ps = cfg["stage1"], cfg["policy"]
    types = s["interaction_types"] + s["level_only_types"]
    rows = []
    for ctr in cfg["countries"]:
        for fuel in s["fuels"]:
            g_all = weekly[(weekly.country == ctr) & (weekly.fuel == fuel)].set_index("date").sort_index()
            g = sample_slice(g_all, s["samples"]["main"])
            lags = s1[(s1.country == ctr) & (s1.fuel == fuel)][["p_lags", "q_lags"]].iloc[0]
            p, q = int(lags.p_lags), int(lags.q_lags)

            _, clean = masks(g, s)                       # the Stage 1 normal-times model
            lr, gap = long_run(g, s, clean, trend=True)
            sr = fit_ecm(ecm_frame(g, gap, s, p, q, "exclude"), clean, s)

            for a, b, name in episodes(g, types):
                stop = min(b + ps["after_weeks"], len(g) - 1)
                cf_pre = simulate(g, s, lr, sr, p, q, a, stop)
                pre_week = g.iloc[max(a - ps["tax_baseline_weeks_before"], 0)]   # taxes before the episode
                for t in range(a, stop + 1):
                    w = g.iloc[t]
                    phase = w.intervention if w.intervention else "after_" + g.iloc[b].intervention.split(";")[-1]
                    market = (cf_pre[t] + w.fixed_taxes_lcu) * (1 + w.vat_pct / 100)
                    full = (cf_pre[t] + pre_week.fixed_taxes_lcu) * (1 + pre_week.vat_pct / 100)
                    rows.append({"level": "weekly", "country": ctr, "fuel": fuel, "date": g.index[t],
                                 "episode": name, "phase": phase,
                                 "episode_start": g.index[a], "episode_end": g.index[b],
                                 "in_force": t <= b,
                                 "pre_tax_actual": w.price_pre_tax_lcu, "pre_tax_cf": cf_pre[t],
                                 "retail_actual": w.price_with_tax_lcu,
                                 "retail_cf_market": market, "retail_cf_full": full})
    out = pd.DataFrame(rows)
    out["market_effect_pct"] = 100 * (out.retail_actual / out.retail_cf_market - 1)
    out["tax_effect_pct"] = 100 * (out.retail_cf_market / out.retail_cf_full - 1)
    out["total_effect_pct"] = 100 * (out.retail_actual / out.retail_cf_full - 1)
    return out


# --- Monthly: fuel inflation and the direct effect on headline ------------------------------------------

def monthly_effects(cfg, wk, monthly):
    rows = []
    for ctr in cfg["countries"]:
        m = monthly[monthly.country == ctr]
        fuels_idx = m[m.component == "fuels"].set_index("month").sort_index()
        wts = {f: m[m.component == f].set_index("month")["weight"] for f in cfg["stage1"]["fuels"]}
        w = wk[wk.country == ctr].copy()
        w["month"] = w.date.dt.to_period("M").dt.to_timestamp()
        ratios = {}
        for kind in ("market", "full"):
            per_fuel = (w.assign(r=w[f"retail_cf_{kind}"] / w.retail_actual)
                        .groupby(["month", "fuel"]).r.mean().unstack())
            fw = pd.DataFrame({f: wts[f].reindex(per_fuel.index) for f in per_fuel.columns}).fillna(1.0)
            ratios[kind] = (per_fuel * fw).sum(axis=1) / fw.where(per_fuel.notna()).sum(axis=1)
        idx = fuels_idx["index"]
        out = pd.DataFrame({"fuels_index_actual": idx, "fuel_weight": fuels_idx["weight"]})
        yoy = lambda s_: 100 * (s_ / s_.shift(12) - 1)
        out["fuels_yoy_actual"] = yoy(idx)
        for kind, r in ratios.items():
            cf_idx = idx * r.reindex(idx.index).fillna(1.0)       # equal to actual outside the episodes
            out[f"fuels_yoy_cf_{kind}"] = yoy(cf_idx)
            out[f"headline_effect_{kind}_pp"] = out.fuel_weight / 1000 * (out.fuels_yoy_actual - out[f"fuels_yoy_cf_{kind}"])
            out[f"price_level_gap_{kind}_pct"] = 100 * (1 / r.reindex(idx.index) - 1)
        active = sorted(set(ratios["market"].index))
        out = out.loc[min(active) - pd.DateOffset(months=12): max(active) + pd.DateOffset(months=12)]
        out = out.reset_index().rename(columns={"month": "date"})
        out["level"], out["country"], out["fuel"] = "monthly", ctr, "all"
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


# --- Entry point (called by run.py) ------------------------------------------------------------

def run(cfg):
    weekly = pd.read_parquet(utils.DATA_PROCESSED / "fuel_weekly.parquet")
    monthly = pd.read_parquet(utils.DATA_PROCESSED / "hicp_monthly.parquet")
    s1 = pd.read_csv(utils.RESULTS / "stage1_passthrough.csv")
    s1 = s1[(s1["sample"] == "main") & (s1.spec == "baseline") & (s1.policy_handling == "interactions")]

    wk = weekly_counterfactuals(cfg, weekly, s1)
    mo = monthly_effects(cfg, wk, monthly)
    res = pd.concat([wk, mo], ignore_index=True)
    res.to_csv(utils.RESULTS / "policy_counterfactual.csv", index=False)
    summarise(wk, mo, cfg)


def summarise(wk, mo, cfg):
    print("\n    Effect on pump prices by measure (average of petrol and diesel, % vs. counterfactual):")
    print(f"    {'':4s}{'measure (phase)':34s}{'weeks':>6s}{'market avg':>11s}{'market peak':>12s}{'taxes avg':>11s}")
    for (ctr, phase), d in wk.groupby(["country", "phase"], sort=False):
        by_week = d.groupby("date")[["market_effect_pct", "tax_effect_pct"]].mean()
        peak = by_week.market_effect_pct.loc[by_week.market_effect_pct.abs().idxmax()]
        print(f"    {ctr:4s}{phase[:33]:34s}{len(by_week):6d}{by_week.market_effect_pct.mean():11.1f}"
              f"{peak:12.1f}{by_week.tax_effect_pct.mean():11.1f}")
    print("    (negative = actual pump price below the counterfactual; 'after_...' = the weeks after a measure ended)")

    print("\n    Direct effect on headline HICP inflation (y/y, pp): largest reduction and largest increase")
    for ctr in cfg["countries"]:
        d = mo[mo.country == ctr].dropna(subset=["headline_effect_full_pp"])
        if d.empty:
            continue
        for label, i in (("lowest", d.headline_effect_full_pp.idxmin()), ("highest", d.headline_effect_full_pp.idxmax())):
            r = d.loc[i]
            print(f"    {ctr} {label:8s}{r.headline_effect_full_pp:6.2f} pp in {r.date:%Y-%m} "
                  f"(market measures {r.headline_effect_market_pp:5.2f}, "
                  f"taxes {r.headline_effect_full_pp - r.headline_effect_market_pp:5.2f})")
    print("\n    Saved outputs/results/policy_counterfactual.csv")