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

Robustness runs on the main sample (`spec`): no_demand_control; gas_control (current and lagged
% change in the EU gas price added: the oil effect holding gas fixed - a lower bound, because before
about 2015 European gas contracts were oil-indexed, so part of gas was itself an oil effect);
constant_taxes (HICP at constant tax rates as the dependent variable, removing all indirect-tax
changes, not just fuel taxes; the fuel tax control is then dropped).

Policy versions (config `policy_handling`), as in Stage 1:
  interactions  main: policy share, its interaction with oil, and the tax-wedge control
  ignore        none of the policy terms and no tax control (naive)
  dummies       policy share as a level control and the tax control, no interaction
  exclude       tax control; windows containing any policy month are dropped

Reports cumulative responses per 10% oil rise (normal times and with a measure in force),
the split of the headline response into contributions of fuels, other energy, food and core
(year-specific weights, each with a band; `contribution_responses`),
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
    out["gas"] = 100 * np.log(c["gas_usd"]).diff()
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


def break_dummy(index, h, s):
    """1 if the window t..t+h contains the 2017 ECOICOP break month (D2)."""
    brk = pd.Timestamp(s["break_month"])
    return ((index <= brk) & (index + pd.DateOffset(months=h) >= brk)).astype(float)


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
    if sample.get("gas"):                      # robustness: oil effect holding the EU gas price fixed
        for L in range(0, s["lags"] + 1):
            d[f"gas_l{L}"] = inp["gas"].shift(L)
    for mth in range(2, 13):
        d[f"m{mth}"] = (d.index.month == mth).astype(float)
    if comp in s["break_components"]:
        d["break_in_window"] = break_dummy(d.index, h, s)

    # policy share over the window t..t+h
    S = inp["policy"].rolling(h + 1).mean().shift(-h)
    if (version in ("interactions", "dummies", "exclude") and comp in s["tax_control_components"]
            and not sample.get("constant_taxes")):    # constant-tax index: tax changes are already removed
        d["tax"] = tax_effect(inp, [comp] if comp in fuels else fuels, h)   # petrol/diesel: own taxes
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
    results, crude_share, oil_path, checks = [], {}, {}, {}

    runs = [("main", "baseline", v, s["samples"]["main"], {}) for v in cfg["policy_handling"]]
    runs += [("main", "no_demand_control", "interactions", s["samples"]["main"], {"demand": False}),
             ("main", "gas_control", "interactions", s["samples"]["main"], {"gas": True}),
             ("main", "constant_taxes", "interactions", s["samples"]["main"], {"constant_taxes": True}),
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
            rows_c = m[m.component == comp].set_index("month").sort_index()
            for sample_name, spec, version, sample, opts in runs:
                price = rows_c["index_ct" if opts.get("constant_taxes") else "index"]
                if "only" in opts and comp not in opts["only"]:
                    continue
                if comp in s["short_components"] and (sample_name, spec) != ("main", "baseline"):
                    continue                                # too short for the sub-samples
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
                    f, fse = res.params["fx"], res.bse["fx"]       # per 10% weakening of the currency vs USD
                    results.append({**base, "regime": "normal", "measure": "fx_response",
                                    "estimate": k * f, "lower": k * (f - z * fse), "upper": k * (f + z * fse)})
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
        results.append({"country": ctr, "component": "pump_price", "sample": "main", "spec": "chain",
                        "policy_handling": "interactions", "main_estimate": False, "horizon": np.nan,
                        "n_obs": np.nan, "regime": "normal", "measure": "crude_share",
                        "estimate": crude_share[ctr], "lower": np.nan, "upper": np.nan})
        rows, checks[ctr] = contribution_responses(m, inp, s, "interactions" if identified else "dummies",
                                                   fuels, ctr)
        results += rows
        note = "" if identified else " (policy effect not estimated: too few policy months, level control only)"
        print(f"    {ctr}: {policy_months} months with a non-tax measure{note}")

    res = pd.DataFrame(results)
    utils.RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(utils.RESULTS / "stage2_responses.csv", index=False)
    summarise(res, cfg, crude_share, oil_path, checks)


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


def contributions(m):
    """Monthly contributions of fuels, energy, food and core to headline HICP inflation, pp.

    The HICP is chain-linked each December with that year's weights w (per mille), so within year y
        I_tot,t / I_tot,Dec(y-1) = sum_c w_c,y/1000 * I_c,t / I_c,Dec(y-1)
    and the contribution of c to the monthly % change of the headline is
        100 * w_c,y/1000 * (I_c,t - I_c,t-1) / I_c,Dec(y-1) * I_tot,Dec(y-1) / I_tot,t-1.
    Energy, food (incl. alcohol and tobacco) and core weights sum to 1000, so their contributions
    add up to the headline change. A year with a missing weight is interpolated from its neighbours.
    Returns the contributions, the actual headline % change, and the interpolated (component, year)s."""
    idx = m.pivot(index="month", columns="component", values="index").sort_index()
    wt = m.pivot(index="month", columns="component", values="weight").sort_index()
    annual = wt.groupby(wt.index.year).first()
    filled = [(c, y) for c in ("fuels", "energy", "food", "core")
              for y in annual.index[annual[c].isna() & annual[c].interpolate(limit_area="inside").notna()]]
    annual = annual.interpolate(limit_area="inside")
    w = annual.reindex(idx.index.year).set_axis(idx.index)
    dec = pd.DatetimeIndex([pd.Timestamp(d.year - 1, 12, 1) for d in idx.index])
    base = idx.reindex(dec).set_axis(idx.index)
    scale = base["headline"] / idx["headline"].shift(1)
    c = pd.DataFrame({k: 100 * w[k] / 1000 * idx[k].diff() / base[k] * scale
                      for k in ("fuels", "energy", "food", "core")})
    return c, 100 * idx["headline"].pct_change(), filled


def contribution_responses(m, inp, s, version, fuels, ctr):
    """Headline response split into contributions (pp), each with its own band (decisions D11).

    Every part is regressed on the same right-hand side as the headline (its lags, oil, fx, controls,
    the 2017 break dummy) on the same months, so the parts add up exactly to `total`. The dependent
    variable is the sum of monthly contributions over t..t+h, i.e. pp of headline inflation."""
    c, pct, filled = contributions(m)
    parts = pd.DataFrame({
        "fuels": c["fuels"], "other_energy": c["energy"] - c["fuels"], "food": c["food"], "core": c["core"],
        "direct": c["energy"], "indirect": c["food"] + c["core"],
        "total": c[["energy", "food", "core"]].sum(axis=1, min_count=3)})
    gap = (parts["total"] - pct).loc[s["samples"]["main"]["start"]:].abs()
    price = m[m.component == "headline"].set_index("month")["index"].sort_index()
    z, k = norm.ppf(0.5 + s["band"] / 2), s["scale"]
    rows = []
    for sample_name in ("main", "pre2021"):
        sample = s["samples"][sample_name]
        for h in range(s["horizons"] + 1):
            d = lp_frame(price, inp, "headline", h, s, version, {}, fuels)
            d["break_in_window"] = break_dummy(d.index, h, s)
            cum = parts.rolling(h + 1).sum().shift(-h)               # sum of contributions over t..t+h
            valid = cum.notna().all(axis=1)
            for part in parts.columns:
                d["y"] = cum[part].where(valid)
                res = fit(d, h, s, sample["start"], sample["end"], version)
                b, se = res.params["oil"], res.bse["oil"]
                rows.append({"country": ctr, "component": "headline", "sample": sample_name,
                             "spec": "contribution", "policy_handling": version, "main_estimate": False,
                             "horizon": h, "n_obs": int(res.nobs), "regime": "normal",
                             "measure": f"contrib_{part}", "estimate": k * b,
                             "lower": k * (b - z * se), "upper": k * (b + z * se)})
    check = {"mean_abs_gap_pp": gap.mean(), "max_abs_gap_pp": gap.max(), "filled_weights": filled}
    return rows, check


def summarise(res, cfg, crude_share, oil_path, checks):
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

    contrib = res[res.spec == "contribution"]
    parts = ["fuels", "other_energy", "food", "core", "total"]
    print(f"\n    Headline response split into contributions, pp per {s['scale']}% oil rise "
          f"(year-specific HICP weights; parts add up to total):")
    print(f"    {'':14s}" + "".join(f"{p:>14s}" for p in parts) + f"{'headline LP':>14s}")
    for ctr in cfg["countries"]:
        for sample_name in ("main", "pre2021"):
            for h in (3, 12):
                c = contrib[(contrib.country == ctr) & (contrib["sample"] == sample_name) & (contrib.horizon == h)]
                c = c.set_index("measure")
                cells = "".join(f"{c.loc[f'contrib_{p}', 'estimate']:14.2f}" for p in parts)
                lp = res[(res.country == ctr) & (res.component == "headline") & (res["sample"] == sample_name)
                         & (res.spec == "baseline") & (res.horizon == h) & (res.regime == "normal")
                         & (res.measure == "response") & (res.policy_handling == c.policy_handling.iloc[0])]
                print(f"    {ctr} {sample_name:8s}{h:2d}m" + cells + f"{lp.estimate.iloc[0]:14.2f}")
    print("    Additivity check, energy + food + core contributions vs. actual headline monthly change "
          "(from 2008):")
    for ctr, chk in checks.items():
        filled = ", ".join(f"{c} {y}" for c, y in chk["filled_weights"]) or "none"
        print(f"    {ctr}: mean abs gap {chk['mean_abs_gap_pp']:.3f} pp, max {chk['max_abs_gap_pp']:.3f} pp; "
              f"weights interpolated: {filled}")

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