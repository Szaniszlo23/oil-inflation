"""
chain.py - the pass-through chain: Stage 1 and Stage 2 combined, crude oil -> headline HICP.

For a 10% rise in Brent (USD), per country, month by month (h = 0..12):

  1. Brent in local currency, %: local projection of Brent in local currency on the oil shock
     (Stage 2 specification), so it includes the oil price's own later moves.
  2. Pre-tax pump price: the Stage 1 normal-times model (weekly, local currency per litre) is fed
     that Brent path week by week. Monthly Brent averages are placed mid-month and interpolated
     linearly to weeks; the weekly pre-tax response is averaged back to months. Expressed as
     the share of the Brent move passed on (1 = one-for-one in local currency per litre).
  3. Pump price with tax, %: pre-tax change x (1 + VAT) / pump price = pass-through x crude share,
     with the average price structure (weeks without a measure) of the Stage 2 sample. Per fuel.
  4. HICP fuels, %: petrol and diesel combined with their HICP weights (predicted), compared with
     the Stage 2 estimate of the HICP fuels response (estimated).
  5. Direct effect of fuels on headline, pp: predicted HICP fuels x average fuel weight, compared
     with the Stage 2 contribution of fuels.
  6. Other energy, food, core and the headline total, pp: Stage 2 contributions.

Two pairings (config `chain.pairs`): normal times (Stage 1 2008-19, Stage 2 2008-21) and the
full period (Stage 1 and Stage 2 2008-26). Predicted values are point estimates (no bands).

Writes outputs/results/chain.csv (long format: pairing, country, horizon, step, source, value).
"""
import numpy as np
import pandas as pd

from pipeline import utils
from pipeline.policy import normal_times_model
from pipeline.stage1 import cumulative_paths
from pipeline.stage2 import fit, lp_frame, monthly_inputs

WEEKS_PER_MONTH = 52 / 12


# --- Step 1: Brent in local currency ------------------------------------------------------------

def brent_path(m, s2, sample, version, fuels):
    """Cumulative % change of Brent in local currency after a 10% oil shock, h = 0..H."""
    inp = monthly_inputs(m, fuels)
    c = m.drop_duplicates("month").set_index("month").sort_index()
    price = np.exp(inp["brent"] / 100) * c["lcu_per_eur"] / c["usd_per_eur"]
    path = []
    for h in range(s2["horizons"] + 1):
        d = lp_frame(price, inp, "brent_lcu", h, s2, version, {}, fuels)
        path.append(s2["scale"] * fit(d, h, s2, sample["start"], sample["end"], version).params["oil"])
    return np.array(path)


# --- Step 2: Stage 1 weekly dynamics applied to the monthly Brent path -------------------------

def stage1_kernel(g_all, s1, sample_name, p, q, weeks):
    """Cumulative pre-tax response to a permanent +1 step in crude (local currency per litre)."""
    lr, sr, _ = normal_times_model(g_all, s1, sample_name, p, q)
    point = {name: np.array([v]) for name, v in sr.params.items()}
    return cumulative_paths(point, np.array([lr.params["x"]]), p, q, weeks)[:, 0]


def monthly_response(brent_pct, kernel):
    """Pre-tax response (in % of the Brent level, local currency) averaged by month.

    Monthly Brent averages are placed at mid-month (month h at h + 0.5, month -1 = 0) and
    interpolated linearly to weeks; the weekly Brent increments are passed through the Stage 1
    kernel and the result is averaged within each month. Returns (pre-tax path, Brent path) by month."""
    H = len(brent_pct) - 1
    t = -0.5 + np.arange(int((H + 1.5) * WEEKS_PER_MONTH) + 1) / WEEKS_PER_MONTH   # months from start of month 0
    level = np.interp(t, np.r_[-0.5, np.arange(H + 1) + 0.5], np.r_[0.0, brent_pct])
    step = np.diff(np.r_[0.0, level])
    k = np.r_[kernel, np.full(max(0, len(t) - len(kernel)), kernel[-1])]
    pre = np.array([step[:w + 1] @ k[w::-1] for w in range(len(t))])                # convolution
    month = np.floor(t).astype(int)
    by_month = lambda x: np.array([x[month == h].mean() for h in range(H + 1)])
    return by_month(pre), by_month(level)


# --- Steps 3-6 ------------------------------------------------------------------------------------

def price_structure(w, start, end):
    """Crude share of the pump price incl. VAT, weeks without a measure in the sample."""
    w = w[(w.date >= start) & (w.date <= (end or w.date.max())) & ~w.in_intervention]
    return ((1 + w.vat_pct / 100) * w.brent_lcu_prev_week).mean() / w.price_with_tax_lcu.mean()


def run(cfg):
    s1, s2, cc = cfg["stage1"], cfg["stage2"], cfg["chain"]
    fuels = list(cfg["series"]["oil_bulletin"]["fuels"])
    weekly = pd.read_parquet(utils.DATA_PROCESSED / "fuel_weekly.parquet")
    monthly = pd.read_parquet(utils.DATA_PROCESSED / "hicp_monthly.parquet")
    r1 = pd.read_csv(utils.RESULTS / "stage1_passthrough.csv")
    r2 = pd.read_csv(utils.RESULTS / "stage2_responses.csv")
    versions = r2[r2.main_estimate].drop_duplicates("country").set_index("country").policy_handling
    weeks = int((s2["horizons"] + 2) * WEEKS_PER_MONTH) + 1

    rows = []
    for pairing, (s1_sample, s2_sample) in cc["pairs"].items():
        smp = s2["samples"][s2_sample]
        start, end = pd.Timestamp(smp["start"]), pd.Timestamp(smp["end"]) if smp["end"] else None
        for ctr in cfg["countries"]:
            m, version = monthly[monthly.country == ctr], versions[ctr]
            add = lambda step, source, values, lower=None, upper=None: rows.extend(
                {"pairing": pairing, "stage1_sample": s1_sample, "stage2_sample": s2_sample, "country": ctr,
                 "horizon": h, "step": step, "source": source, "value": float(v),
                 "lower": np.nan if lower is None else float(lower[h]),
                 "upper": np.nan if upper is None else float(upper[h])} for h, v in enumerate(values))

            brent = brent_path(m, s2, smp, version, fuels)
            add("brent_lcu_pct", "stage2", brent)

            # petrol / diesel shares of the fuel weight in the sample (HICP weights exist from 2015)
            in_smp = m[(m.month >= start) & (m.month <= (end or m.month.max()))]
            fw = {f: in_smp[in_smp.component == f].drop_duplicates("month").weight.mean() for f in fuels}
            share = {f: fw[f] / sum(fw.values()) for f in fuels}
            pump = np.zeros(len(brent))
            for fuel in fuels:
                g_all = weekly[(weekly.country == ctr) & (weekly.fuel == fuel)].set_index("date").sort_index()
                lags = r1[(r1.country == ctr) & (r1.fuel == fuel)][["p_lags", "q_lags"]].iloc[0]
                kernel = stage1_kernel(g_all, s1, s1_sample, int(lags.p_lags), int(lags.q_lags), weeks)
                pre, level = monthly_response(brent, kernel)
                crude_share = price_structure(weekly[(weekly.country == ctr) & (weekly.fuel == fuel)], start, end)
                add(f"pretax_passthrough_{fuel}", "stage1", pre / level)
                add(f"crude_share_{fuel}", "data", np.full(len(brent), crude_share))
                add(f"pump_{fuel}_pct", "stage1", pre * crude_share)
                pump += share[fuel] * pre * crude_share
            add("hicp_fuels_pct", "predicted", pump)

            est = r2[(r2.country == ctr) & (r2.component == "fuels") & (r2["sample"] == s2_sample)
                     & (r2.spec == "baseline") & (r2.regime == "normal") & (r2.measure == "response")
                     & (r2.policy_handling == version)].sort_values("horizon")
            add("hicp_fuels_pct", "stage2", est.estimate.values, est.lower.values, est.upper.values)

            fuel_weight = in_smp[in_smp.component == "fuels"].drop_duplicates("month").weight.mean() / 1000
            add("direct_fuels_pp", "predicted", pump * fuel_weight)
            con = r2[(r2.country == ctr) & (r2.spec == "contribution") & (r2["sample"] == s2_sample)]
            for part in ("fuels", "other_energy", "food", "core", "total"):
                d = con[con.measure == f"contrib_{part}"].sort_values("horizon")
                step = "direct_fuels_pp" if part == "fuels" else f"{part}_pp"
                add(step, "stage2", d.estimate.values, d.lower.values, d.upper.values)

    out = pd.DataFrame(rows)
    out.to_csv(utils.RESULTS / "chain.csv", index=False)
    summarise(out, cfg)


def summarise(out, cfg):
    def v(pairing, ctr, step, source, h):
        d = out[(out.pairing == pairing) & (out.country == ctr) & (out.step == step) & (out.source == source)
                & (out.horizon == h)]
        return d.value.iloc[0]

    for pairing in cfg["chain"]["pairs"]:
        print(f"\n    Pass-through chain, {pairing} (per {cfg['stage2']['scale']}% oil rise): Brent in local "
              "currency % -> pre-tax pass-through (Stage 1) -> pump price % -> HICP fuels % predicted vs "
              "estimated -> fuels in headline pp predicted vs estimated -> headline pp")
        for ctr in cfg["countries"]:
            cells = []
            for h in (1, 3, 12):
                pt = np.mean([v(pairing, ctr, f"pretax_passthrough_{f}", "stage1", h) for f in cfg["stage1"]["fuels"]])
                cells.append(f"{h}m {v(pairing, ctr, 'brent_lcu_pct', 'stage2', h):5.1f} -> {pt:4.2f} -> "
                             f"{v(pairing, ctr, 'hicp_fuels_pct', 'predicted', h):4.1f} vs "
                             f"{v(pairing, ctr, 'hicp_fuels_pct', 'stage2', h):4.1f} -> "
                             f"{v(pairing, ctr, 'direct_fuels_pp', 'predicted', h):4.2f} vs "
                             f"{v(pairing, ctr, 'direct_fuels_pp', 'stage2', h):4.2f} -> "
                             f"{v(pairing, ctr, 'total_pp', 'stage2', h):4.2f}")
            print(f"    {ctr}: " + "\n        ".join(cells))
    print("\n    Saved outputs/results/chain.csv")
