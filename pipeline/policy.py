"""
policy.py - what pump prices and inflation would have been without the interventions.

For every episode with a non-tax measure (price cap, maximum price, margin cap, discount,
Hungary's post-cap window), per country and fuel:

  1. Market price without the measure: the Stage 1 normal-times model (long-run relation
     and weekly dynamics, estimated on market weeks only) is started from the actual
     pre-tax price in the week before the episode and fed the actual Brent path, week by
     week, until `after_weeks` weeks after the episode ends.
     The model is estimated on `policy.model_sample` (default main). A crude-only model has no
     refining margins, so after 2020 the counterfactual misses their rise: positive 'after_...'
     effects in 2026 are mostly crack spreads, not policy. (Estimating on pre2020 makes this
     worse: it under-predicts every post-2020 price.)
  2. Two counterfactual pump prices:
       market  counterfactual pre-tax price + the taxes actually charged
               -> isolates the cap / margin cap / discount
       full    counterfactual pre-tax price + VAT and per-litre taxes as they were
               `tax_baseline_weeks_before` weeks before the episode (tax cuts often started a few
               days before the price measures) -> adds the effect of tax changes made around it
  3. Monthly: the HICP fuels index tracks pump prices almost one-for-one (Stage 2 chain),
     so the gap becomes counterfactual fuel inflation and, with the fuel weight, a direct
     effect on headline inflation (pp). Second-round effects are not included.

Cross-check without a model (`policy.crosscheck` in config.yaml): the treated country's pre-tax
margin over crude (EUR/l) minus the average margin of control countries, relative to a baseline
window before the measure. Control-country weeks with their own measure in force (or within
`exclude_after_weeks` after one) are left out, e.g. Orlen's pre-election pricing in Poland in 2023.
Valid for Hungary 2021-23 with Poland and Romania as controls: their 2022 measures were tax cuts
(not in the pre-tax price) and a discount not reported to the Oil Bulletin. Not valid for 2026,
when all three countries intervened.

Writes outputs/results/policy_counterfactual.csv (long format, `level` = weekly / monthly)
and outputs/results/policy_crosscheck.csv (monthly).
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

def simulate(g, s, lr, sr, p, q, start, stop, trend_offset=0):
    """Pre-tax price path from week `start` to `stop` under normal-times dynamics, actual Brent.
    trend_offset: weeks between the start of the model's estimation sample and the start of g."""
    y = g[s["y"]].to_numpy(dtype=float)
    x = g[s["x"]].to_numpy(dtype=float)
    trend = (np.arange(len(g)) + trend_offset) / 52.0
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


def normal_times_model(g_all, s, model_sample, p, q):
    """Stage 1 normal-times model (long run + weekly dynamics) estimated on clean weeks of a sample."""
    g_est = sample_slice(g_all, s["samples"][model_sample])
    _, clean = masks(g_est, s)
    lr, gap = long_run(g_est, s, clean, trend=True)
    sr = fit_ecm(ecm_frame(g_est, gap, s, p, q, "exclude"), clean, s)
    return lr, sr, g_est.index[0]


def weekly_counterfactuals(cfg, weekly, s1, model_sample):
    s, ps = cfg["stage1"], cfg["policy"]
    types = s["interaction_types"] + s["level_only_types"]
    rows = []
    for ctr in cfg["countries"]:
        for fuel in s["fuels"]:
            g_all = weekly[(weekly.country == ctr) & (weekly.fuel == fuel)].set_index("date").sort_index()
            g = sample_slice(g_all, s["samples"]["main"])
            lags = s1[(s1.country == ctr) & (s1.fuel == fuel)][["p_lags", "q_lags"]].iloc[0]
            p, q = int(lags.p_lags), int(lags.q_lags)

            lr, sr, est_start = normal_times_model(g_all, s, model_sample, p, q)
            offset = int(round((g.index[0] - est_start).days / 7))      # weekly grid

            blocks = episodes(g, types)
            for i, (a, b, name) in enumerate(blocks):
                # the after-window stops before the next episode starts (no week simulated twice)
                next_start = blocks[i + 1][0] if i + 1 < len(blocks) else len(g)
                stop = min(b + ps["after_weeks"], next_start - 1, len(g) - 1)
                cf_pre = simulate(g, s, lr, sr, p, q, a, stop, trend_offset=offset)
                pre_week = g.iloc[max(a - ps["tax_baseline_weeks_before"], 0)]   # taxes before the episode
                for t in range(a, stop + 1):
                    w = g.iloc[t]
                    phase = w.intervention if w.intervention else "after_" + g.iloc[b].intervention.split(";")[-1]
                    market = (cf_pre[t] + w.fixed_taxes_lcu) * (1 + w.vat_pct / 100)
                    full = (cf_pre[t] + pre_week.fixed_taxes_lcu) * (1 + pre_week.vat_pct / 100)
                    rows.append({"level": "weekly", "model_sample": model_sample, "country": ctr, "fuel": fuel,
                                 "date": g.index[t], "episode": name, "phase": phase,
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
        out["model_sample"] = wk.model_sample.iloc[0]
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


# --- All interventions since 2021: caps and tax changes on one timeline -------------------------------

def intervention_timeline(cfg, weekly, wk, monthly):
    """Direct effect of all fuel interventions on headline inflation (y/y, pp), monthly from `timeline_start`.

    Two counterfactual pump prices per week:
      no caps          pre-tax price without the cap / margin cap / discount (model counterfactual inside
                       each episode and its after-window, actual otherwise) + the taxes actually charged
      no intervention  the same pre-tax price + VAT and per-litre taxes frozen at their level in the first
                       week of `timeline_start`
    Caps = actual vs 'no caps'; tax changes = the rest (every VAT and excise change after the start date,
    cuts and increases, including rule-based excise changes such as Hungary's Brent-linked excise).
    As in monthly_effects(): fuel weight x (actual - counterfactual annual change of the fuels index)."""
    start = pd.Timestamp(cfg["policy"]["timeline_start"])
    cap_pre = wk[["country", "fuel", "date", "pre_tax_cf"]].drop_duplicates(["country", "fuel", "date"])
    rows = []
    for ctr in cfg["countries"]:
        m = monthly[monthly.country == ctr]
        fuels_idx = m[m.component == "fuels"].set_index("month").sort_index()
        wts = {f: m[m.component == f].set_index("month")["weight"] for f in cfg["stage1"]["fuels"]}
        ratios = {"caps": {}, "none": {}}
        for fuel in cfg["stage1"]["fuels"]:
            g = weekly[(weekly.country == ctr) & (weekly.fuel == fuel) & (weekly.date >= start)].sort_values("date")
            g = g.merge(cap_pre[(cap_pre.country == ctr) & (cap_pre.fuel == fuel)].drop(columns=["country", "fuel"]),
                        on="date", how="left")
            pre = g.pre_tax_cf.fillna(g.price_pre_tax_lcu)
            base = g.iloc[0]
            cf_caps = (pre + g.fixed_taxes_lcu) * (1 + g.vat_pct / 100)
            cf_none = (pre + base.fixed_taxes_lcu) * (1 + base.vat_pct / 100)
            month = g.date.dt.to_period("M").dt.to_timestamp()
            ratios["caps"][fuel] = (cf_caps / g.price_with_tax_lcu).groupby(month).mean()
            ratios["none"][fuel] = (cf_none / g.price_with_tax_lcu).groupby(month).mean()
        idx = fuels_idx["index"]
        yoy = lambda s_: 100 * (s_ / s_.shift(12) - 1)
        out = pd.DataFrame({"fuel_weight": fuels_idx["weight"], "fuels_yoy_actual": yoy(idx)})
        for kind, per_fuel in ratios.items():
            pf = pd.DataFrame(per_fuel)
            fw = pd.DataFrame({f: wts[f].reindex(pf.index) for f in pf.columns}).fillna(1.0)
            r = ((pf * fw).sum(axis=1) / fw.where(pf.notna()).sum(axis=1)).reindex(idx.index).fillna(1.0)
            out[f"fuels_yoy_cf_{kind}"] = yoy(idx * r)
            out[f"price_gap_{kind}_pct"] = 100 * (1 / r - 1)
        out["caps_pp"] = out.fuel_weight / 1000 * (out.fuels_yoy_actual - out.fuels_yoy_cf_caps)
        out["total_pp"] = out.fuel_weight / 1000 * (out.fuels_yoy_actual - out.fuels_yoy_cf_none)
        out["taxes_pp"] = out.total_pp - out.caps_pp
        out = out.loc[start:].reset_index().rename(columns={"month": "date"})
        out["country"] = ctr
        rows.append(out)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(utils.RESULTS / "intervention_timeline.csv", index=False)
    return out


def summarise_timeline(tl, cfg):
    print("\n    All interventions since " + cfg["policy"]["timeline_start"][:7] +
          ": direct effect on headline inflation (y/y, pp), annual average (caps / tax changes / total)")
    for ctr in cfg["countries"]:
        d = tl[tl.country == ctr].set_index("date")
        a = d[["caps_pp", "taxes_pp", "total_pp"]].groupby(d.index.year).mean()
        print(f"    {ctr}: " + "  ".join(f"{y}: {r.caps_pp:+.2f}/{r.taxes_pp:+.2f}/{r.total_pp:+.2f}"
                                       for y, r in a.iterrows()))
        lo, hi = d.total_pp.idxmin(), d.total_pp.idxmax()
        print(f"        lowest {d.total_pp[lo]:+.2f} pp in {lo:%Y-%m}, highest {d.total_pp[hi]:+.2f} pp in {hi:%Y-%m}")
    print("    Saved outputs/results/intervention_timeline.csv")


# --- Cross-check: treated country vs. control countries ----------------------------------------------

def crosscheck(cfg, weekly, monthly):
    """Market effect of a measure from the margin over crude relative to control countries (no model).
    effect_pct = 100 * (actual retail price / counterfactual - 1), counterfactual = actual + the margin
    gap (with VAT); the same definition as market_effect_pct in the model counterfactual."""
    s, cc = cfg["stage1"], cfg["policy"]["crosscheck"]
    w = weekly.copy()
    w["margin_eur"] = (w[s["y"]] - w[s["x"]]) / w["lcu_per_eur"]
    clean = pd.concat([masks(g.sort_values("date").set_index("date"), s)[1].rename("clean").to_frame()
                       .assign(country=c, fuel=f) for (c, f), g in w.groupby(["country", "fuel"])])
    w = w.merge(clean.reset_index(), on=["date", "country", "fuel"], how="left")
    ctrl = (w[w.country.isin(cc["controls"]) & w.clean]
            .groupby(["date", "fuel"]).margin_eur.mean().rename("control_margin_eur"))
    t = (w[w.country == cc["country"]].merge(ctrl.reset_index(), on=["date", "fuel"], how="left")
         .sort_values("date"))
    t["diff_eur"] = t.margin_eur - t.control_margin_eur
    base = t[t.date.between(*cc["baseline"])].groupby("fuel").diff_eur.mean()
    t["gap_eur"] = t.diff_eur - t.fuel.map(base)
    t["retail_cf"] = t.price_with_tax_lcu - t.gap_eur * t.lcu_per_eur * (1 + t.vat_pct / 100)
    t["effect_pct"] = 100 * (t.price_with_tax_lcu / t.retail_cf - 1)
    t = t[t.date.between(*cc["window"])]
    t["month"] = t.date.dt.to_period("M").dt.to_timestamp()
    mo = t.pivot_table(index="month", columns="fuel", values=["gap_eur", "effect_pct"])
    mo.columns = [f"{v}_{f}" for v, f in mo.columns]
    mo["effect_pct_avg"] = mo[[c for c in mo.columns if c.startswith("effect_pct_")]].mean(axis=1)
    f = monthly[(monthly.country == cc["country"]) & (monthly.component == "fuels")].set_index("month").sort_index()
    idx = f["index"]
    cf_idx = idx / (1 + mo.effect_pct_avg.reindex(idx.index).fillna(0) / 100)   # equal to actual outside the window
    # direct effect on headline y/y inflation, as in monthly_effects(): includes base effects after the cap
    yoy = lambda s_: 100 * (s_ / s_.shift(12) - 1)
    effect = f["weight"] / 1000 * (yoy(idx) - yoy(cf_idx))
    mo["fuel_weight"] = f["weight"].reindex(mo.index)
    mo["headline_effect_pp"] = effect.reindex(mo.index)
    mo = mo.reset_index().rename(columns={"month": "date"})
    mo["country"], mo["controls"] = cc["country"], "+".join(cc["controls"])
    mo.to_csv(utils.RESULTS / "policy_crosscheck.csv", index=False)
    return mo


# --- Entry point (called by run.py) ------------------------------------------------------------

def run(cfg):
    weekly = pd.read_parquet(utils.DATA_PROCESSED / "fuel_weekly.parquet")
    monthly = pd.read_parquet(utils.DATA_PROCESSED / "hicp_monthly.parquet")
    s1 = pd.read_csv(utils.RESULTS / "stage1_passthrough.csv")
    s1 = s1[(s1["sample"] == "main") & (s1.spec == "baseline") & (s1.policy_handling == "interactions")]
    model_sample = cfg["policy"].get("model_sample", "main")

    wk = weekly_counterfactuals(cfg, weekly, s1, model_sample)
    mo = monthly_effects(cfg, wk, monthly)
    pd.concat([wk, mo], ignore_index=True).to_csv(utils.RESULTS / "policy_counterfactual.csv", index=False)
    summarise(wk, mo, cfg, "policy_counterfactual.csv")
    summarise_timeline(intervention_timeline(cfg, weekly, wk, monthly), cfg)

    if "crosscheck" in cfg["policy"]:
        cc = crosscheck(cfg, weekly, monthly)
        summarise_crosscheck(cc, mo, cfg)
    else:
        print("    note: no policy.crosscheck in config.yaml - cross-check skipped")


def summarise(wk, mo, cfg, fname):
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
                  f"taxes {r.headline_effect_full_pp - r.headline_effect_market_pp:5.2f}; "
                  f"fuel weight {r.fuel_weight:.1f}, fuel inflation actual {r.fuels_yoy_actual:.1f}% "
                  f"vs counterfactual {r.fuels_yoy_cf_full:.1f}%)")
    print(f"\n    Saved outputs/results/{fname}")


def summarise_crosscheck(cc, mo, cfg):
    c = cfg["policy"]["crosscheck"]
    m = mo[mo.country == c["country"]].set_index(pd.to_datetime(mo[mo.country == c["country"]].date))
    print(f"\n    Cross-check, {c['country']} vs. {'+'.join(c['controls'])} (margin over crude, baseline "
          f"{c['baseline'][0]} to {c['baseline'][1]}): pump price vs. counterfactual, %, monthly")
    print(f"    {'month':10s}{'petrol':>9s}{'diesel':>9s}{'avg':>8s}{'model avg':>11s}{'headline y/y, pp':>18s}")
    for _, r in cc.iterrows():
        d = pd.Timestamp(r.date)
        model = m["price_level_gap_market_pct"].get(d, np.nan)
        if d.month % 3 == 0:
            print(f"    {d:%Y-%m}   {r.get('effect_pct_petrol', np.nan):9.1f}{r.get('effect_pct_diesel', np.nan):9.1f}"
                  f"{r.effect_pct_avg:8.1f}{model:11.1f}{r.headline_effect_pp:18.2f}")
    print("    (model avg = Stage 1 counterfactual, same definition; headline y/y = direct effect, cross-check;"
          " quarterly rows shown)")
    print("    Saved outputs/results/policy_crosscheck.csv")