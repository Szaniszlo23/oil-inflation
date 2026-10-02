"""
stage2.py - oil prices -> consumer prices (HICP), monthly local projections.

For each country, HICP component and horizon h = 0..H (docs/decisions.md D20):

  100*(ln P_{t+h} - ln P_{t-1}) = a_h + b_h*oil_t + f_h*fx_t
                                  + c_h*oil_t*S_{t,h} + d_h*S_{t,h}      (policy terms)
                                  + g_h*tax_{t,h}                        (fuel tax changes, fuels/energy/headline)
                                  + lags of the component, oil, fx (and euro-area industrial production)
                                  + month dummies (+ 2017 break dummy) + e_{t+h}

  oil_t   monthly % change in Brent (USD); responses are scaled to a 10% rise
  fx_t    monthly % change in local currency per USD
  S_{t,h} share of days from t to t+h with a non-tax measure in force (caps, margin caps, discounts)
  tax     % change in the fuel price caused by tax changes alone between t-1 and t+h, holding
          the pre-tax price fixed: ln((pre_{t-1} + F_{t+h})(1 + v_{t+h})) - ln((pre_{t-1} + F_{t-1})(1 + v_{t-1})),
          F = excise + other per-litre taxes, v = VAT, average of petrol and diesel. (The ratio of the
          prices with and without tax would be a bad control: it falls mechanically when oil raises
          the pre-tax price.)

Policy versions (config `policy_handling`), as in Stage 1:
  interactions  main: policy share, its interaction with oil, and the tax-wedge control
  ignore        none of the policy terms and no tax control (naive)
  dummies       policy share as a level control and the tax control, no interaction
  exclude       tax control; windows containing any policy month are dropped

Reports cumulative responses per 10% oil rise (normal times and with a measure in force),
the split of the headline response into energy (of which fuels) and indirect effects,
and the pass-through chain (Brent in local currency -> pump price -> HICP fuels) as a check
on Stage 1. Writes outputs/results/stage2_responses.csv.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from pipeline import utils


def monthly_inputs(m, fuels):
    """Country-level monthly series (the same for every component row)."""
    c = m.drop_duplicates("month").set_index("month").sort_index()
    out = pd.DataFrame(index=c.index)
    out["brent"] = 100 * np.log(c["brent_usd"])
    out["oil"] = out["brent"].diff()
    out["fx"] = 100 * np.log(c["lcu_per_eur"] / c["usd_per_eur"]).diff()     # local currency per USD
    out["ip"] = 100 * np.log(c["ea_ip"]).diff()
    out["policy"] = c["intervention_share"]
    for f in fuels:
        for col in ("pre_tax_lcu", "fixed_taxes_lcu", "vat_pct", "with_tax_lcu"):
            out[f"{f}_{col}"] = c[f"{f}_{col}"]
    out["brent_lcu"] = c["brent_lcu"]
    return out


def tax_effect(inp, fuels, h):
    """% change in the fuel price from tax changes alone, t-1 -> t+h, pre-tax price held at t-1."""
    parts = []
    for f in fuels:
        pre0 = inp[f"{f}_pre_tax_lcu"].shift(1)
        F0, v0 = inp[f"{f}_fixed_taxes_lcu"].shift(1), inp[f"{f}_vat_pct"].shift(1)
        F1, v1 = inp[f"{f}_fixed_taxes_lcu"].shift(-h), inp[f"{f}_vat_pct"].shift(-h)
        parts.append(100 * (np.log((pre0 + F1) * (1 + v1 / 100)) - np.log((pre0 + F0) * (1 + v0 / 100))))
    return pd.concat(parts, axis=1).mean(axis=1, skipna=False)


def lp_frame(price, inp, comp, h, s, version, sample, fuels):
    """Regression data for one component and horizon."""
    lnp = 100 * np.log(price)
    d = pd.DataFrame(index=price.index)
    d["y"] = lnp.shift(-h) - lnp.shift(1)
    d["const"] = 1.0
    d["oil"] = inp["oil"]
    d["fx"] = inp["fx"]
    own = lnp.diff()
    for L in range(1, s["lags"] + 1):
        if comp != "brent_lcu":                # its own change is exactly oil + fx
            d[f"own_l{L}"] = own.shift(L)
        d[f"oil_l{L}"] = inp["oil"].shift(L)
        d[f"fx_l{L}"] = inp["fx"].shift(L)
    if comp in s["demand_control_components"] and sample.get("demand", True):
        for L in range(0, s["lags"] + 1):
            d[f"ip_l{L}"] = inp["ip"].shift(L)
    for mth in range(2, 13):
        d[f"m{mth}"] = (d.index.month == mth).astype(float)
    if comp in s["break_components"]:
        brk = pd.Timestamp(s["break_month"])
        d["break_in_window"] = ((d.index <= brk) & (d.index + pd.DateOffset(months=h) >= brk)).astype(float)

    # policy share over the window t..t+h
    S = inp["policy"].rolling(h + 1).mean().shift(-h)
    if version in ("interactions", "dummies", "exclude") and comp in s["tax_control_components"]:
        d["tax"] = tax_effect(inp, fuels, h)
    if version == "interactions":
        d["policy"] = S
        d["oil_x_policy"] = inp["oil"] * S
    elif version == "dummies":
        d["policy"] = S
    d["_policy_window"] = S
    return d


def fit(d, h, s, start, end, version):
    d = d.loc[start:end] if end else d.loc[start:]
    rows = d.drop(columns="_policy_window").notna().all(axis=1)
    if end:                                    # the whole window t..t+h must lie inside the sample
        rows &= (d.index + pd.DateOffset(months=h)) <= pd.Timestamp(end)
    if version == "exclude":
        rows &= d["_policy_window"].fillna(0) == 0
    data = d[rows].drop(columns="_policy_window")
    X = data.drop(columns="y")
    X = X.loc[:, [c for c in X.columns if c == "const" or X[c].abs().sum() > 0]]
    return sm.OLS(data["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})


def run(cfg):
    s = cfg["stage2"]
    z = norm.ppf(0.5 + s["band"] / 2)
    k = s["scale"]
    monthly = pd.read_parquet(utils.DATA_PROCESSED / "hicp_monthly.parquet")
    fuels = list(cfg["series"]["oil_bulletin"]["fuels"])
    results, crude_share, oil_path = [], {}, {}

    runs = [("main", "baseline", v, s["samples"]["main"], {}) for v in cfg["policy_handling"]]
    runs += [("main", "no_demand_control", "interactions", s["samples"]["main"], {"demand": False}),
             ("pre2021", "baseline", "interactions", s["samples"]["pre2021"], {}),
             ("from2017", "baseline", "interactions", s["samples"]["from2017"], {"only": s["break_components"]})]

    for ctr in cfg["countries"]:
        m = monthly[monthly.country == ctr]
        inp = monthly_inputs(m, fuels)
        main_start = s["samples"]["main"]["start"]
        policy_months = int((inp.loc[main_start:, "policy"] > 0).sum())
        identified = policy_months >= s["min_policy_months"]   # enough months to estimate a policy effect
        crude_share[ctr], oil_path[ctr] = benchmark_inputs(inp, fuels, s, cfg)
        for comp in s["components"]:
            price = m[m.component == comp].set_index("month")["index"].sort_index()
            for sample_name, spec, version, sample, opts in runs:
                if "only" in opts and comp not in opts["only"]:
                    continue
                if version == "interactions" and not identified:
                    if (sample_name, spec) == ("main", "baseline"):
                        continue                            # the `dummies` run is the main estimate here
                    version = "dummies"                     # too few policy months for an interaction
                for h in range(s["horizons"] + 1):
                    d = lp_frame(price, inp, comp, h, s, version, opts, fuels)
                    res = fit(d, h, s, sample["start"], sample["end"], version)
                    base = {"country": ctr, "component": comp, "sample": sample_name, "spec": spec,
                            "policy_handling": version, "horizon": h, "n_obs": int(res.nobs),
                            "main_estimate": sample_name == "main" and spec == "baseline"
                                             and version == ("interactions" if identified else "dummies")}
                    b, se = res.params["oil"], res.bse["oil"]
                    results.append({**base, "regime": "normal", "measure": "response",
                                    "estimate": k * b, "lower": k * (b - z * se), "upper": k * (b + z * se)})
                    if "oil_x_policy" in res.params:
                        c = res.params["oil_x_policy"]
                        cov = res.cov_params()
                        se_sum = np.sqrt(cov.loc["oil", "oil"] + cov.loc["oil_x_policy", "oil_x_policy"]
                                         + 2 * cov.loc["oil", "oil_x_policy"])
                        results.append({**base, "regime": "policy_in_force", "measure": "response",
                                        "estimate": k * (b + c), "lower": k * (b + c - z * se_sum),
                                        "upper": k * (b + c + z * se_sum)})
                        sc = res.bse["oil_x_policy"]
                        results.append({**base, "regime": "policy_effect", "measure": "response",
                                        "estimate": k * c, "lower": k * (c - z * sc), "upper": k * (c + z * sc)})
        chain_series = {
            "brent_lcu": np.exp(inp["brent"] / 100) * m.drop_duplicates("month").set_index("month").sort_index()["lcu_per_eur"]
                         / m.drop_duplicates("month").set_index("month").sort_index()["usd_per_eur"],
            "pump_price": inp[[f"{f}_with_tax_lcu" for f in fuels]].mean(axis=1, skipna=False),
        }
        for comp, price in chain_series.items():
            for h in range(s["horizons"] + 1):
                d = lp_frame(price, inp, "fuels" if comp == "pump_price" else comp, h, s, "interactions", {}, fuels)
                res = fit(d, h, s, s["samples"]["main"]["start"], s["samples"]["main"]["end"], "interactions")
                b, se = res.params["oil"], res.bse["oil"]
                results.append({"country": ctr, "component": comp, "sample": "main", "spec": "chain",
                                "policy_handling": "interactions", "main_estimate": False,
                                "horizon": h, "n_obs": int(res.nobs),
                                "regime": "normal", "measure": "response", "estimate": k * b,
                                "lower": k * (b - z * se), "upper": k * (b + z * se)})
        note = "" if identified else " (policy effect not estimated: too few policy months, level control only)"
        print(f"    {ctr}: {policy_months} months with a non-tax measure{note}")

    res = pd.DataFrame(results)
    res = pd.concat([res, decomposition(res, monthly, cfg)], ignore_index=True)
    utils.RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(utils.RESULTS / "stage2_responses.csv", index=False)
    summarise(res, cfg, crude_share, oil_path)


def benchmark_inputs(inp, fuels, s, cfg):
    """For the Stage 1 cross-check: average crude share of the retail fuel price (VAT included)
    outside policy months, and Brent's own cumulative response to its shock (oil momentum)."""
    start = s["samples"]["main"]["start"]
    x = inp.loc[start:]
    shares = [((1 + x[f"{f}_vat_pct"] / 100) * x["brent_lcu"] / x[f"{f}_with_tax_lcu"])[x["policy"] == 0]
              for f in fuels]
    share = float(pd.concat(shares).mean())
    path = {}
    for h in range(s["horizons"] + 1):
        d = pd.DataFrame({"y": x["brent"].shift(-h) - x["brent"].shift(1), "const": 1.0, "oil": x["oil"]})
        for L in range(1, s["lags"] + 1):
            d[f"oil_l{L}"] = x["oil"].shift(L)
        d = d.dropna()
        path[h] = sm.OLS(d["y"], d.drop(columns="y")).fit().params["oil"]
    return share, path


def decomposition(res, monthly, cfg):
    """Headline response = energy weight x energy response (of which fuels) + indirect remainder.
    Uses the latest year's HICP weights (per mille)."""
    main = res[(res["sample"] == "main") & (res.spec == "baseline") & (res.regime == "normal")
               & (res.measure == "response")]
    rows = []
    for ctr in cfg["countries"]:
        m = monthly[monthly.country == ctr]
        latest = m[m.month == m.month.max()].set_index("component")["weight"]
        resp = main[(main.country == ctr) & main.main_estimate].pivot(index="horizon", columns="component",
                                                                       values="estimate")
        energy = latest["energy"] / 1000 * resp["energy"]
        fuels = latest["fuels"] / 1000 * resp["fuels"]
        parts = {"energy_direct": energy, "fuels_direct": fuels, "indirect": resp["headline"] - energy}
        for name, series in parts.items():
            for h, v in series.items():
                rows.append({"country": ctr, "component": "headline", "sample": "main", "spec": "baseline",
                             "policy_handling": "main", "main_estimate": True, "horizon": h, "regime": "normal",
                             "measure": name, "estimate": v, "lower": np.nan, "upper": np.nan, "n_obs": np.nan})
    return pd.DataFrame(rows)


def summarise(res, cfg, crude_share, oil_path):
    s = cfg["stage2"]
    main = res[(res["sample"] == "main") & (res.spec == "baseline")]

    def val(ctr, comp, h, version="interactions", regime="normal", measure="response"):
        d = main[(main.country == ctr) & (main.component == comp) & (main.horizon == h)
                 & (main.regime == regime) & (main.measure == measure)]
        d = d[d.policy_handling == version] if version != "interactions" else d[d.main_estimate]
        return d.iloc[0] if len(d) else None

    print(f"\n    Response to a {s['scale']}% oil price rise, % (main estimate, normal times, "
          f"{int(s['band'] * 100)}% band):")
    print(f"    {'':4s}{'headline 3m':>22s}{'headline 12m':>22s}{'fuels 2m':>22s}{'core 12m':>22s}")
    for ctr in cfg["countries"]:
        cells = []
        for comp, h in (("headline", 3), ("headline", 12), ("fuels", 2), ("core", 12)):
            r = val(ctr, comp, h)
            cells.append(f"{r.estimate:6.2f} [{r.lower:5.2f},{r.upper:5.2f}]")
        print(f"    {ctr:4s}" + "".join(f"{c:>22s}" for c in cells))

    print("\n    Headline response at 12 months, split (pp):")
    for ctr in cfg["countries"]:
        e, f, i = (val(ctr, "headline", 12, measure=x).estimate for x in ("energy_direct", "fuels_direct", "indirect"))
        print(f"    {ctr}: energy {e:.2f} (of which fuels {f:.2f}), indirect {i:.2f}")

    # pass-through chain at 1-3 months: Brent in local currency -> pump price -> HICP fuels
    chain = res[(res["sample"] == "main") & (res.spec == "chain")]
    if len(chain):
        print(f"\n    Pass-through chain, response to a {s['scale']}% oil rise (%): Brent in local currency"
              " -> pump price with tax -> HICP fuels   [crude share x Brent = full pass-through benchmark]")
        for ctr in cfg["countries"]:
            cells = []
            for h in (1, 2, 3):
                b_ = chain[(chain.country == ctr) & (chain.component == "brent_lcu") & (chain.horizon == h)].estimate.iloc[0]
                p_ = chain[(chain.country == ctr) & (chain.component == "pump_price") & (chain.horizon == h)].estimate.iloc[0]
                f_ = val(ctr, "fuels", h).estimate
                cells.append(f"{h}m {b_:5.1f} -> {p_:4.1f} -> {f_:4.1f} [{crude_share[ctr] * b_:4.1f}]")
            print(f"    {ctr}: " + "   ".join(cells))

    eff = main[(main.regime == "policy_effect") & (main.component.isin(["fuels", "headline"]))
               & (main.horizon.isin([2, 12]))]
    if len(eff):
        print("\n    Effect of non-tax measures on the response to a 10% oil rise (pp, where estimable):")
        for _, r in eff.iterrows():
            print(f"    {r.country} {r.component:9s}{r.horizon:3.0f}m {r.estimate:6.2f} [{r.lower:.2f}, {r.upper:.2f}]")

    print("\n    Checks:")
    ok = True
    for ctr in cfg["countries"]:
        for comp in ("headline", "fuels"):
            for h in (3, 12):
                a, b = val(ctr, comp, h), val(ctr, comp, h, version="exclude")
                if b is not None and abs(a.estimate - b.estimate) > s["sanity"]["max_gap_normal_vs_exclude_pp"]:
                    ok = False
                    print(f"    {ctr} {comp} {h}m: normal times {a.estimate:.2f} vs excluded version {b.estimate:.2f}")
    if ok:
        print("    normal-times responses match the version without policy periods")
    print(f"\n    Saved {len(res):,} rows to outputs/results/stage2_responses.csv")