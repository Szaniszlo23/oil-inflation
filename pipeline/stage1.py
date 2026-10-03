"""
stage1.py - crude oil -> pump prices before tax, weekly (docs/decisions.md D18).

For each country and fuel, a two-step error-correction model in levels
(local currency per litre):

  Step 1  Long-run market relation, dynamic OLS on clean weeks only
          (no measure in force and not within `exclude_after_weeks` after one):
              y_t = a + b*t + beta*x_t + sum_j c_j*dx_{t+j} + u_t,      j = -L..L
          beta = long-run pass-through; gap_t = y_t - (a + b*t + beta*x_t).

  Step 2a Normal-times weekly dynamics, on clean weeks:
              dy_t = c + alpha*gap_{t-1} + sum_i g_i*dy_{t-i} + sum_j psi_j*dx_{t-j} + e_t

  Step 2b Policy effects, on the weeks with a measure in force (and the weeks just after),
          measured against the normal-times model from 2a:
              dy_t - predicted_t = sum_k [lam_k*D_k + alpha_k*D_k*gap_{t-1} + psi_k*D_k*dx_t]
                                   + lam_post*D_post + e_t
          k = policy types in `interaction_types` (hard cap, margin cap, discount).
          All weeks are used; the market dynamics are not contaminated by capped weeks.

Policy versions (config `policy_handling`):
  interactions  main estimate, as above (steps 1, 2a, 2b)
  ignore        no policy terms; long run on all weeks
  dummies       level dummies only; long run on all weeks
  exclude       steps 1 and 2a only (equals `interactions` in normal times, without the policy effects)

Reports long-run pass-through, speed, weeks to 50% and 90% of the long-run effect,
impact and cumulative pass-through
by week (normal times and under each policy type), the retail-price elasticity,
and diagnostics. Writes outputs/results/stage1_passthrough.csv (long format).

Diagnostics - is there a long-run link? (per country x fuel, samples pre2020 and main, clean weeks):
  margin tests    m_t = y_t - x_t (beta fixed at 1, theory: pre-tax = crude + margins);
                  ADF (const; const+trend; H0 unit root), KPSS (const; H0 stationary)
  Engle-Granger   static OLS residuals, const and const+trend, all clean weeks (H0 no cointegration)
  ARDL bounds     Pesaran-Shin-Smith case 3, H0 no levels relationship; valid if y, x are I(0) or I(1)
  DL differences  dy_t = c + sum_{j=0..H} b_j dx_{t-j} + e_t, no error-correction term, so it needs
                  no cointegration; cumulative pass-through at each horizon = sum b_0..b_h.
                  Only weeks whose whole window t-H-1..t is clean.
Writes outputs/results/stage1_diagnostics.csv (long format).
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.ardl import UECM, ardl_select_order
from statsmodels.tsa.stattools import adfuller, coint, kpss

from pipeline import utils


# --- Sample and masks -----------------------------------------------------------------

def sample_slice(g, sample):
    start = pd.Timestamp(sample["start"]) if sample["start"] else g.index.min()
    end = pd.Timestamp(sample["end"]) if sample["end"] else g.index.max()
    return g.loc[start:end]


def masks(g, s):
    """Policy-in-force indicator and the 'clean' mask (no measure now or in the last N weeks)."""
    types = s["interaction_types"] + s["level_only_types"]
    in_force = g[[f"policy_{t}" for t in types]].any(axis=1)
    recent = in_force.copy()
    for k in range(1, s["exclude_after_weeks"] + 1):
        recent |= in_force.shift(k, fill_value=False)
    return in_force, ~recent


def no_interpolation_window(g, s):
    """True where neither this week nor any of its lags is an interpolated holiday week."""
    touched = g["interpolated"].copy()
    for k in range(1, s["max_lags"] + 2):
        touched |= g["interpolated"].shift(k, fill_value=False)
    return ~touched


def drop_empty(X, keep=("const",)):
    """Remove regressors that are zero everywhere in the estimation sample (absent policy types)."""
    return X.loc[:, [c for c in X.columns if c in keep or (X[c] != 0).any()]]


# --- Step 1: long-run relation ------------------------------------------------------------

def long_run(g, s, rows, trend, level_dummies=None):
    y, x, L = g[s["y"]], g[s["x"]], s["dols_leads_lags"]
    X = pd.DataFrame({"const": 1.0, "x": x}, index=g.index)
    if trend:
        X["trend"] = np.arange(len(g)) / 52.0          # per year
    dx = x.diff()
    for j in range(-L, L + 1):
        X[f"dx_{j}"] = dx.shift(-j)                     # j > 0: leads, j < 0: lags
    if level_dummies is not None:
        X = X.join(level_dummies.astype(float))
    data = pd.concat([y.rename("y"), X], axis=1)[rows].dropna()
    Xd = drop_empty(data.drop(columns="y"))
    res = sm.OLS(data["y"], Xd).fit(cov_type="HAC", cov_kwds={"maxlags": s["hac_lags"]})

    fitted = res.params["const"] + res.params["x"] * x
    if trend:
        fitted = fitted + res.params["trend"] * X["trend"]
    if level_dummies is not None:
        for c in level_dummies.columns:
            if c in res.params:
                fitted = fitted + res.params[c] * level_dummies[c].astype(float)
    return res, y - fitted


# --- Step 2: short-run dynamics ---------------------------------------------------------------

def ecm_frame(g, gap, s, p, q, version):
    y, x = g[s["y"]], g[s["x"]]
    d = pd.DataFrame(index=g.index)
    d["dy"] = y.diff()
    d["const"] = 1.0
    d["gap1"] = gap.shift(1)
    for i in range(1, p + 1):
        d[f"dy{i}"] = d["dy"].shift(i)
    d["dx0"] = x.diff()
    for j in range(1, q + 1):
        d[f"dx{j}"] = d["dx0"].shift(j)
    if version == "dummies":
        for k in s["interaction_types"] + s["level_only_types"]:
            d[f"lev_{k}"] = g[f"policy_{k}"].astype(float)
    return d


def policy_frame(g, d, s):
    """Policy terms for step 2b: level, speed and impact shifts per type, level-only types."""
    z = pd.DataFrame(index=g.index)
    for k in s["interaction_types"]:
        D = g[f"policy_{k}"].astype(float)
        z[f"lev_{k}"] = D
        z[f"gap1_{k}"] = D * d["gap1"]
        z[f"dx0_{k}"] = D * d["dx0"]
    for k in s["level_only_types"]:
        z[f"lev_{k}"] = g[f"policy_{k}"].astype(float)
    return z


def fit_policy(g, d, normal, rows, s):
    """Step 2b: regress the part of dy the normal-times model cannot explain on the policy terms."""
    X = d[normal.params.index]
    surprise = (d["dy"] - X.mul(normal.params, axis=1).sum(axis=1, min_count=len(normal.params))).rename("r")
    data = pd.concat([surprise, policy_frame(g, d, s)], axis=1)[rows].dropna()
    Z = drop_empty(data.drop(columns="r"), keep=())
    if Z.shape[1] == 0:
        return None
    return sm.OLS(data["r"], Z).fit(cov_type="HAC", cov_kwds={"maxlags": s["hac_lags"]})


def fit_ecm(d, rows, s, robust=True):
    data = d[rows].dropna()
    X = drop_empty(data.drop(columns="dy"))
    model = sm.OLS(data["dy"], X)
    if robust:
        return model.fit(cov_type="HAC", cov_kwds={"maxlags": s["hac_lags"]})
    return model.fit()


def select_lags(g, gap, s, rows):
    """AIC over p, q = 0..max_lags on a common estimation sample."""
    m = s["max_lags"]
    common = rows & (np.arange(len(g)) > m + 1)
    best = None
    for p in range(m + 1):
        for q in range(m + 1):
            res = fit_ecm(ecm_frame(g, gap, s, p, q, "exclude"), common, s, robust=False)
            if best is None or res.aic < best[0]:
                best = (res.aic, p, q)
    return best[1], best[2]


# --- Responses ------------------------------------------------------------------------------

def cumulative_paths(b, beta, p, q, H, k=None):
    """Price response to a permanent +1 move in crude (local currency per litre), weeks 0..H.
    b: dict of coefficient arrays (one value per draw); beta: array; k: policy type or None."""
    n = len(beta)
    alpha = b["gap1"] + (b[f"gap1_{k}"] if k else 0.0)
    psi = [b["dx0"] + (b[f"dx0_{k}"] if k else 0.0)] + [b[f"dx{j}"] for j in range(1, q + 1)]
    gam = [b[f"dy{i}"] for i in range(1, p + 1)]
    level = np.zeros(n)
    dys, out = [], []
    for t in range(H + 1):
        gap_prev = level - beta * (1.0 if t >= 1 else 0.0)
        dy = alpha * gap_prev
        if t <= q:
            dy = dy + psi[t]
        for i, gi in enumerate(gam, start=1):
            if t - i >= 0:
                dy = dy + gi * dys[t - i]
        dys.append(dy)
        level = level + dy
        out.append(level.copy())
    return np.array(out)                                 # shape (H+1, draws)


def weeks_to(paths, beta, share):
    """First week in which the cumulative response reaches `share` of long-run pass-through
    (inf if not within the horizon). paths: (H+1, draws); beta: (draws,)."""
    reached = paths >= share * beta[None, :]
    first = np.argmax(reached, axis=0).astype(float)
    first[~reached.any(axis=0)] = np.inf
    return first


def band(draws):
    draws = np.asarray(draws, dtype=float)
    method = "nearest" if np.isinf(draws).any() else "linear"
    return (np.nanpercentile(draws, 2.5, method=method), np.nanpercentile(draws, 97.5, method=method))


# --- One estimation -------------------------------------------------------------------------

def estimate(g, s, version, p, q, rng, trend=True, drop_interpolated=False):
    in_force, clean = masks(g, s)
    all_rows = pd.Series(True, index=g.index)
    if drop_interpolated:
        all_rows &= no_interpolation_window(g, s)
        clean &= all_rows

    types = s["interaction_types"] + s["level_only_types"]
    level = g[[f"policy_{t}" for t in types]].rename(columns=lambda c: c.replace("policy_", "lr_"))
    if version in ("interactions", "exclude"):
        lr, gap = long_run(g, s, clean, trend)
    elif version == "dummies":
        lr, gap = long_run(g, s, all_rows, trend, level_dummies=level)
    else:
        lr, gap = long_run(g, s, all_rows, trend)

    d = ecm_frame(g, gap, s, p, q, version)
    if version in ("interactions", "exclude"):
        sr = fit_ecm(d, clean, s)                                  # step 2a: market weeks
        pol = fit_policy(g, d, sr, ~clean & all_rows, s) if version == "interactions" else None
    else:
        sr, pol = fit_ecm(d, all_rows, s), None

    # simulated coefficient draws for the bands (normal-times and policy blocks drawn separately)
    n = s["draws"]
    params = sr.params.copy()
    draws = rng.multivariate_normal(sr.params.values, sr.cov_params().values, size=n)
    b = {name: draws[:, i] for i, name in enumerate(sr.params.index)}
    if pol is not None:
        params = pd.concat([params, pol.params])
        pdraws = rng.multivariate_normal(pol.params.values, pol.cov_params().values, size=n)
        b.update({name: pdraws[:, i] for i, name in enumerate(pol.params.index)})
    beta_hat, beta_se = lr.params["x"], lr.bse["x"]
    beta = rng.normal(beta_hat, beta_se, size=n)
    point = {name: np.array([v]) for name, v in params.items()}
    H = s["max_horizon_weeks"]

    out = []

    def add(regime, measure, est, lo=np.nan, hi=np.nan, horizon=np.nan):
        out.append({"regime": regime, "measure": measure, "horizon": horizon,
                    "estimate": float(est), "lower": float(lo), "upper": float(hi)})

    add("normal", "long_run", beta_hat, beta_hat - 1.96 * beta_se, beta_hat + 1.96 * beta_se)
    regimes = [("normal", None)] + [(k, k) for k in s["interaction_types"] if f"gap1_{k}" in params]
    for regime, k in regimes:
        a_draw = b["gap1"] + (b[f"gap1_{k}"] if k else 0)
        a_pt = params["gap1"] + (params[f"gap1_{k}"] if k else 0)
        add(regime, "speed", a_pt, *band(a_draw))
        i_draw = b["dx0"] + (b[f"dx0_{k}"] if k else 0)
        i_pt = params["dx0"] + (params[f"dx0_{k}"] if k else 0)
        add(regime, "impact", i_pt, *band(i_draw))
        paths = cumulative_paths(b, beta, p, q, H, k)
        path_pt = cumulative_paths(point, np.array([beta_hat]), p, q, H, k)
        for share in (0.5, 0.9):
            add(regime, f"weeks_to_{int(share * 100)}pct",
                weeks_to(path_pt, np.array([beta_hat]), share)[0], *band(weeks_to(paths, beta, share)))
        for h in s["horizons"]:
            add(regime, "cumulative", path_pt[h, 0], *band(paths[h]), horizon=h)
    for k in types:
        if f"lev_{k}" in params:
            add(k, "level_effect_per_week", params[f"lev_{k}"], *band(b[f"lev_{k}"]))

    # retail-price elasticity: % change in the price with tax per 1% change in crude
    recent = g[clean].tail(s["elasticity_weeks"])
    ratio = ((1 + recent["vat_pct"] / 100) * recent[s["x"]]).mean() / recent["price_with_tax_lcu"].mean()
    add("normal", "retail_elasticity", beta_hat * ratio, *band(beta * ratio))

    # diagnostics
    add("all", "r2", sr.rsquared)
    add("all", "ecm_t", sr.tvalues["gap1"])                     # t-stat on the speed of adjustment
    add("all", "ljung_box_p",
        acorr_ljungbox(sr.resid, lags=[s["hac_lags"]], return_df=True)["lb_pvalue"].iloc[0])
    first_policy = clean[~clean].index.min() if (~clean).any() else None
    stretch = g.loc[g.index < first_policy] if first_policy is not None else g
    if len(stretch) >= s["min_clean_weeks_for_coint"]:
        add("all", "engle_granger_p",
            coint(stretch[s["y"]], stretch[s["x"]], trend="ct" if trend else "c")[1])

    for r in out:
        r["n_obs"] = int(sr.nobs) + (int(pol.nobs) if pol is not None else 0)
    return out


# --- Diagnostics: is there a long-run link? ---------------------------------------------------

def _plain(fn, *args, **kw):
    """Plain-tuple output on every statsmodels version (0.15 warns without result_object)."""
    try:
        return fn(*args, result_object=False, **kw)
    except TypeError:                                     # older versions: no such argument
        return fn(*args, **kw)


def margin_tests(margin):
    """Stationarity of the margin y - x (beta fixed at 1)."""
    m = margin.dropna().to_numpy()
    out = []
    for reg in ("c", "ct"):
        stat, pv, *_ = _plain(adfuller, m, regression=reg, autolag="AIC")
        out.append({"test": f"margin_adf_{reg}", "statistic": stat, "p_value": pv, "h0": "unit root"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)   # KPSS p-values are capped at 0.01-0.10
        for reg in ("c", "ct"):
            stat, pv, *_ = _plain(kpss, m, regression=reg, nlags="auto")
            out.append({"test": f"margin_kpss_{reg}", "statistic": stat, "p_value": pv, "h0": "stationary"})
    return out


def engle_granger(y, x):
    out = []
    for trend in ("c", "ct"):
        stat, pv, _ = coint(y.to_numpy(), x.to_numpy(), trend=trend)
        out.append({"test": f"engle_granger_{trend}", "statistic": stat, "p_value": pv,
                    "h0": "no cointegration"})
    return out


def bounds_test(y, x, max_lag):
    """ARDL bounds test, case 3 (unrestricted constant, no trend). Lags by AIC, at least 1 each."""
    y = y.rename("y").reset_index(drop=True)
    X = x.rename("x").reset_index(drop=True).to_frame()
    sel = ardl_select_order(y, max_lag, X, max_lag, trend="c", ic="aic")
    p = max(len(sel.model.ar_lags or []), 1)
    q = max((sel.model.dl_lags.get("x") or [0])[-1], 1)
    res = UECM(y, p, X, q, trend="c").fit()
    bt = res.bounds_test(case=3)
    stat = getattr(bt, "statistic", getattr(bt, "stat", np.nan))      # name differs by version
    pv = getattr(bt, "pvalue", getattr(bt, "p_values", None))
    return [{"test": "ardl_bounds_case3", "statistic": stat,
             "p_value": pv["upper"],                      # conservative: I(1) bound
             "p_value_lower_bound": pv["lower"], "h0": "no levels relationship",
             "long_run": -res.params["x.L1"] / res.params["y.L1"], "lags": f"p={p},q={q}"}]


def dl_passthrough(y, x, clean, horizons, hac_lags):
    """Cumulative pass-through from a distributed lag in differences (95% band, as elsewhere)."""
    H = max(horizons)
    dy, dx = y.diff(), x.diff()
    X = pd.concat({f"dx_l{j}": dx.shift(j) for j in range(H + 1)}, axis=1)
    ok = clean.astype(int).rolling(H + 2).min().eq(1)    # whole lag window clean
    d = pd.concat([dy.rename("dy"), X], axis=1)[ok].dropna()
    res = sm.OLS(d["dy"], sm.add_constant(d.drop(columns="dy"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    out = []
    for h in horizons:
        w = np.array([1.0 if c.startswith("dx_l") and int(c[4:]) <= h else 0.0 for c in res.params.index])
        est = float(w @ res.params.to_numpy())
        se = float(np.sqrt(w @ res.cov_params().to_numpy() @ w))
        out.append({"test": "dl_cumulative", "horizon": h, "statistic": est,
                    "lower": est - 1.96 * se, "upper": est + 1.96 * se})
    return out


def diagnostics(g_all, s):
    rows = []
    for name in ("pre2020", "main"):
        g = sample_slice(g_all, s["samples"][name])
        _, clean = masks(g, s)
        y, x = g[s["y"]], g[s["x"]]
        yc, xc = y[clean], x[clean]
        meta = {"sample": name, "clean_weeks": int(clean.sum()),
                "clean_segments": int((clean & ~clean.shift(fill_value=False)).sum())}
        tests = (margin_tests(yc - xc) + engle_granger(yc, xc) + bounds_test(yc, xc, s["max_lags"])
                 + dl_passthrough(y, x, clean, s["horizons"], s["hac_lags"]))
        rows += [{**meta, **t} for t in tests]
    return rows


def summarise_diagnostics(diag):
    lvl = diag[diag.test != "dl_cumulative"].pivot_table(
        index=["country", "fuel", "sample"], columns="test", values="p_value").round(3)
    print("\n    Diagnostics, p-values (clean weeks; margin = pre-tax - crude, beta fixed at 1;"
          " KPSS H0 = stationary, capped at 0.10):")
    print("    " + lvl.to_string().replace("\n", "\n    "))
    dl = diag[(diag.test == "dl_cumulative") & (diag.horizon == 4)].set_index(
        ["country", "fuel", "sample"])[["statistic", "lower", "upper"]].round(2)
    print("\n    Pass-through after 4 weeks, distributed lag in differences (no cointegration needed), 95% band:")
    print("    " + dl.to_string().replace("\n", "\n    "))
    print(f"\n    Saved {len(diag):,} rows to outputs/results/stage1_diagnostics.csv")


# --- Entry point (called by run.py) -------------------------------------------------------------

def run(cfg):
    s = cfg["stage1"]
    rng = np.random.default_rng(s["seed"])
    weekly = pd.read_parquet(utils.DATA_PROCESSED / "fuel_weekly.parquet")
    results, diag = [], []

    for ctr in cfg["countries"]:
        for fuel in s["fuels"]:
            g_all = weekly[(weekly.country == ctr) & (weekly.fuel == fuel)].set_index("date").sort_index()
            g_main = sample_slice(g_all, s["samples"]["main"])
            _, clean = masks(g_main, s)
            _, gap = long_run(g_main, s, clean, trend=True)
            p, q = select_lags(g_main, gap, s, pd.Series(True, index=g_main.index))

            runs = [("main", "baseline", v, g_main, {}) for v in cfg["policy_handling"]]
            runs += [("main", "no_trend", "interactions", g_main, {"trend": False}),
                     ("main", "no_interpolated", "interactions", g_main, {"drop_interpolated": True}),
                     ("pre2020", "baseline", "interactions", sample_slice(g_all, s["samples"]["pre2020"]), {})]
            g_full = sample_slice(g_all, s["samples"]["full"])
            if g_full.index.min() < g_main.index.min():
                runs.append(("full", "baseline", "interactions", g_full, {}))

            for sample, spec, version, g, kw in runs:
                for r in estimate(g, s, version, p, q, rng, **kw):
                    results.append({"country": ctr, "fuel": fuel, "sample": sample, "spec": spec,
                                    "policy_handling": version, "p_lags": p, "q_lags": q, **r})
            diag += [{"country": ctr, "fuel": fuel, **r} for r in diagnostics(g_all, s)]
            print(f"    {ctr} {fuel}: lags dy={p}, dx={q}")

    res = pd.DataFrame(results)
    utils.RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(utils.RESULTS / "stage1_passthrough.csv", index=False)
    summarise(res, s)

    diag = pd.DataFrame(diag)
    diag.to_csv(utils.RESULTS / "stage1_diagnostics.csv", index=False)
    summarise_diagnostics(diag)


def summarise(res, s):
    main = res[(res["sample"] == "main") & (res.spec == "baseline")]

    def get(ctr, fuel, version, regime, measure, h=None):
        d = main[(main.country == ctr) & (main.fuel == fuel) & (main.policy_handling == version)
                 & (main.regime == regime) & (main.measure == measure)]
        d = d[d.horizon == h] if h is not None else d[d.horizon.isna()]
        return d.iloc[0]

    print("\n    Main estimate (interactions), normal times:")
    print(f"    {'':12s}{'long run [95% band]':>22s}{'wks to 50%':>12s}{'wks to 90%':>12s}"
          f"{'after 4 wks':>13s}{'retail elast.':>15s}{'ignoring policy, 4 wks':>24s}")
    lim = s["sanity"]
    for (ctr, fuel), _ in main.groupby(["country", "fuel"], sort=False):
        lr = get(ctr, fuel, "interactions", "normal", "long_run")
        w50 = get(ctr, fuel, "interactions", "normal", "weeks_to_50pct")
        w90 = get(ctr, fuel, "interactions", "normal", "weeks_to_90pct")
        c4 = get(ctr, fuel, "interactions", "normal", "cumulative", 4)
        el = get(ctr, fuel, "interactions", "normal", "retail_elasticity")
        ig = get(ctr, fuel, "ignore", "normal", "cumulative", 4)
        print(f"    {ctr} {fuel:8s}{lr.estimate:9.2f} [{lr.lower:.2f}, {lr.upper:.2f}]{w50.estimate:12.0f}"
              f"{w90.estimate:12.0f}{c4.estimate:13.2f}{el.estimate:15.2f}{ig.estimate:24.2f}")
        if not lim["long_run_min"] <= lr.estimate <= lim["long_run_max"]:
            print("      check: long-run pass-through outside the expected range")
        if not w90.estimate <= lim["max_weeks_to_90pct"]:
            print("      check: slow adjustment to the long run")

    caps = main[(main.policy_handling == "interactions") & (main.measure == "cumulative")
                & (main.horizon == 4) & (~main.regime.isin(["normal", "all"]))]
    if len(caps):
        print("\n    Cumulative pass-through after 4 weeks while a measure is in force:")
        for _, r in caps.iterrows():
            print(f"    {r.country} {r.fuel:8s}{r.regime:12s}{r.estimate:6.2f} [{r.lower:.2f}, {r.upper:.2f}]")
    print(f"\n    Saved {len(res):,} rows to outputs/results/stage1_passthrough.csv")