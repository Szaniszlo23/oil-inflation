"""
build.py - turn the raw downloads into two model-ready tables.

Reads the latest snapshot in data/raw/<date>/, config.yaml and interventions.csv.
Writes:
  data/processed/fuel_weekly.parquet   one row per country x fuel x Oil Bulletin week
  data/processed/hicp_monthly.parquet  one row per country x HICP component x month

Conversions (see docs/decisions.md):
  - Oil Bulletin prices: EUR per 1000 litres -> local currency per litre, with the
    ECB reference rate of the same day (D4: this is the rate the bulletin itself uses)
  - Brent: USD per barrel -> local currency per litre, daily, then per bulletin week
    (same day and previous-week average) and per month
  - Taxes: VAT, excise and other indirect taxes from the bulletin's own tax sheets (D7),
    checked against the bulletin's prices; where they disagree, the rates embedded in
    the prices are used and the periods are reported (D17)
  - Weekly grid: holiday gaps in the bulletin are filled by linear interpolation and
    flagged (`interpolated`); Brent for those Mondays comes from the daily data (D6)
  - Policy flags: price caps, subsidies etc. from interventions.csv, one column per type

Units: "lcu" = local currency (PLN, RON, HUF); all prices and taxes are per litre.
The script stops with an error if any sanity check fails.
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from pipeline import utils


# --- Generic helpers --------------------------------------------------------------

def asof(series, dates):
    """Value of `series` on each date, or its most recent earlier value.
    Returns a frame indexed by date with the value and the date it came from."""
    right = series.dropna().sort_index()
    right = pd.DataFrame({"value": right.values, "source_date": right.index})
    left = pd.DataFrame({"date": pd.DatetimeIndex(dates)}).sort_values("date")
    out = pd.merge_asof(left, right, left_on="date", right_on="source_date", direction="backward")
    return out.set_index("date")


def jsonstat_long(js):
    """Eurostat JSON-stat -> long table with value and status flag per observation."""
    dims = js["id"]
    categories = []
    for d in dims:
        idx = js["dimension"][d]["category"]["index"]
        categories.append(sorted(idx, key=idx.get) if isinstance(idx, dict) else list(idx))
    full = pd.MultiIndex.from_product(categories, names=dims)
    value = pd.Series({int(k): v for k, v in js.get("value", {}).items()}, dtype=float)
    status = pd.Series({int(k): v for k, v in js.get("status", {}).items()}, dtype=object)
    out = pd.DataFrame({"value": value.reindex(range(len(full))).values,
                        "status": status.reindex(range(len(full))).values}, index=full)
    return out.dropna(subset=["value"]).reset_index()


# --- Raw sources ------------------------------------------------------------------

def load_fx(snap):
    raw = pd.read_csv(snap / "ecb_fx.csv", usecols=["CURRENCY", "TIME_PERIOD", "OBS_VALUE"])
    raw = raw.dropna(subset=["OBS_VALUE"])  # days without a fixing (decisions D10)
    fx = raw.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE")
    fx.index = pd.to_datetime(fx.index)
    return fx.sort_index()  # units of currency per EUR


def load_brent(snap, cfg):
    code = cfg["series"]["brent"]["code"]
    obs = json.loads((snap / f"fred_{code}.json").read_text())["observations"]
    s = pd.Series({pd.Timestamp(o["date"]): o["value"] for o in obs})
    return pd.to_numeric(s, errors="coerce").dropna().sort_index()  # USD per barrel; "." = no quote


def brent_lcu_per_litre(brent, fx, currency):
    """Daily Brent in local currency per litre: USD/bbl / (USD per EUR) * (LCU per EUR) / litres."""
    usd = asof(fx["USD"], brent.index)["value"].values
    lcu = asof(fx[currency], brent.index)["value"].values
    return pd.Series(brent.values / usd * lcu / utils.LITRES_PER_BARREL, index=brent.index)


def load_bulletin_prices(snap, cfg):
    """Both price sheets -> long table in EUR per litre."""
    ob = cfg["series"]["oil_bulletin"]
    path = snap / "oil_bulletin_history.xlsx"
    parts = []
    for kind, out_col in (("with_tax", "price_with_tax_eur"), ("without_tax", "price_pre_tax_eur")):
        raw = pd.read_excel(path, sheet_name=ob["sheets"][kind], header=None)
        header = raw.iloc[0].tolist()
        first_col = raw.iloc[:, 0]
        is_date = first_col.map(lambda v: isinstance(v, (datetime, pd.Timestamp)))
        dates = pd.to_datetime(first_col.where(is_date))
        body = raw[is_date].copy()              # data rows are the ones with a date in column A
        body.columns = header
        body.index = dates[dates.notna()]
        for ctr in cfg["countries"]:
            for fuel, product in ob["fuels"].items():
                col = ob["columns"][kind].format(ctr=ctr, fuel=product)
                if col not in body.columns:
                    raise KeyError(f"Column '{col}' not found in sheet '{ob['sheets'][kind]}'")
                s = pd.to_numeric(body[col], errors="coerce") / utils.LITRES_PER_KL
                parts.append(pd.DataFrame({"date": s.index, "country": ctr, "fuel": fuel,
                                           "kind": out_col, "value": s.values}))
    long = pd.concat(parts)
    wide = long.pivot_table(index=["country", "fuel", "date"], columns="kind",
                            values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide.dropna(subset=["price_with_tax_eur", "price_pre_tax_eur"], how="all")


def load_tax_steps(snap, cfg, sheet_key):
    """A tax sheet -> (country, fuel, since, value). Blank cells mean 'no change for this fuel'."""
    ob = cfg["series"]["oil_bulletin"]
    raw = pd.read_excel(snap / "oil_bulletin_history.xlsx", sheet_name=ob["sheets"][sheet_key], header=None)
    since_row = raw.index[raw.iloc[:, 1].astype(str).str.strip() == "Since:"][0]
    labels = raw.iloc[since_row].astype(str).str.strip()
    body = raw.iloc[since_row + 1:]
    country = body.iloc[:, 0].ffill().astype(str).str.rstrip("_")
    since = pd.to_datetime(body.iloc[:, 1], errors="coerce")
    parts = []
    for fuel, prefix in ob["tax_labels"].items():
        col = labels.index[labels.str.startswith(prefix)][0]
        parts.append(pd.DataFrame({"country": country, "fuel": fuel, "since": since,
                                   "value": pd.to_numeric(body.iloc[:, col], errors="coerce")}))
    steps = pd.concat(parts).dropna(subset=["since", "value"])
    steps = steps[steps.country.isin(cfg["countries"])]
    return steps.sort_values("since").drop_duplicates(["country", "fuel", "since"], keep="last")


def tax_on_dates(steps, country, fuel, dates):
    s = steps[(steps.country == country) & (steps.fuel == fuel)]
    if s.empty:
        return pd.Series(np.nan, index=pd.DatetimeIndex(dates))
    return asof(s.set_index("since")["value"], dates)["value"]


def effective_taxes(g, valid_vat, b):
    """Tax rates as embedded in the bulletin's own prices (decisions D17).

    The price sheets are internally consistent, but the dates in the VAT and excise
    sheets are sometimes wrong. So:
      - VAT: the sheet rate, unless the prices clearly imply a different legal rate
        (gap > vat_mismatch_pp and the implied rate within vat_snap_pp of a rate the
        country has used); then that rate. If they disagree but no legal rate fits
        (e.g. a transition week), the previous week's rate is carried forward.
      - fixed_taxes_lcu (excise + other per-litre taxes): always from the prices,
        price with tax / (1 + VAT) - price without tax.
      - tax_change: VAT changed, or fixed taxes moved by more than tax_change_threshold_pct.
    """
    fixed_sheet = g.excise_lcu_sheet + g.other_taxes_lcu_sheet.fillna(0)
    implied_vat = (g.price_with_tax_lcu / (g.price_pre_tax_lcu + fixed_sheet) - 1) * 100
    nearest = implied_vat.apply(lambda v: min(valid_vat, key=lambda r: abs(r - v)) if pd.notna(v) else np.nan)
    mismatch = (implied_vat - g.vat_pct_sheet).abs() > b["vat_mismatch_pp"]
    snaps = (implied_vat - nearest).abs() <= b["vat_snap_pp"]
    use_implied = mismatch & snaps
    # sheet consistent -> sheet rate; clear mismatch -> the legal rate the prices imply;
    # unclear mismatch (e.g. a transition week) -> carry the previous week's rate forward
    vat = g.vat_pct_sheet.where(~mismatch, np.nan).where(~use_implied, nearest)
    vat[g.interpolated] = np.nan                    # interpolated weeks carry the previous rate
    g["vat_pct"] = vat.ffill().fillna(g.vat_pct_sheet)
    g["fixed_taxes_lcu"] = g.price_with_tax_lcu / (1 + g.vat_pct / 100) - g.price_pre_tax_lcu

    fixed_gap = (g.fixed_taxes_lcu / fixed_sheet - 1).abs() * 100
    g["tax_sheet_mismatch"] = (mismatch | (fixed_gap > b["fixed_tax_mismatch_pct"])) & ~g.interpolated

    vat_changed = g.vat_pct.ne(g.vat_pct.shift()) & g.vat_pct.shift().notna()
    fixed_moved = (g.fixed_taxes_lcu.pct_change().abs() * 100) > b["tax_change_threshold_pct"]
    g["tax_change"] = vat_changed | fixed_moved
    return g


def report_tax_mismatches(weekly):
    """Print the periods where the bulletin's tax sheets disagree with its prices."""
    bad = weekly[weekly.tax_sheet_mismatch].sort_values(["country", "fuel", "date"])
    if bad.empty:
        print("    tax sheets consistent with prices everywhere")
        return
    print("    tax sheets disagree with prices (rates taken from the prices instead, decisions D17):")
    for (ctr, fuel), g in bad.groupby(["country", "fuel"]):
        runs = (g.date.diff().dt.days > 21).cumsum()
        spans = [f"{r.date.min():%Y-%m-%d}..{r.date.max():%Y-%m-%d}" if len(r) > 1 else f"{r.date.min():%Y-%m-%d}"
                 for _, r in g.groupby(runs)]
        print(f"      {ctr} {fuel}: {len(g)} weeks - " + ", ".join(spans))


def load_interventions(cfg):
    cols = ["country", "fuel", "measure", "type", "start", "end", "size", "unit", "source"]
    path = utils.INTERVENTIONS_PATH
    if not path.exists() or path.stat().st_size == 0:
        print("    note: interventions.csv is empty - no policy flags")
        return pd.DataFrame(columns=cols)
    iv = pd.read_csv(path, parse_dates=["start", "end"])
    missing = set(cols) - set(iv.columns)
    if missing:
        raise ValueError(f"interventions.csv is missing columns: {sorted(missing)}")
    unknown = set(iv["type"]) - set(cfg["policy_types"])
    if unknown:
        raise ValueError(f"interventions.csv has types not listed in config.yaml policy_types: {sorted(unknown)}")
    iv["fuel"] = iv["fuel"].str.split(";")
    return iv.explode("fuel")


def load_hicp(snap, cfg):
    idx_cfg = cfg["series"]["hicp_index"]
    code_to_name = {code: name for name, code in idx_cfg["components"].items()}
    frames = []
    for ctr in cfg["countries"]:
        js = json.loads((snap / f"eurostat_{idx_cfg['dataset']}_{ctr}.json").read_text())
        df = jsonstat_long(js)
        df = df[df.coicop18.isin(code_to_name)]
        frames.append(df)
    hicp = pd.concat(frames).rename(columns={"geo": "country", "coicop18": "coicop", "value": "index"})
    hicp["component"] = hicp.coicop.map(code_to_name)
    hicp["month"] = pd.PeriodIndex(hicp.time, freq="M").to_timestamp()
    return hicp[["country", "month", "component", "coicop", "index", "status"]]


def load_weights(snap, cfg):
    w_cfg = cfg["series"]["hicp_weights"]
    codes = set(cfg["series"]["hicp_index"]["components"].values())
    js = json.loads((snap / f"eurostat_{w_cfg['dataset']}.json").read_text())
    df = jsonstat_long(js)
    df = df[df.coicop18.isin(codes)]
    df = df.rename(columns={"geo": "country", "coicop18": "coicop", "value": "weight"})
    df["year"] = df.time.astype(int)
    return df[["country", "coicop", "year", "weight"]]


# --- Tables -------------------------------------------------------------------------

def build_fuel_weekly(snap, cfg, fx, brent, iv):
    prices = load_bulletin_prices(snap, cfg)
    taxes = {k: load_tax_steps(snap, cfg, k) for k in ("vat", "excise", "other_taxes")}
    rows = []
    for (ctr, fuel), g in prices.groupby(["country", "fuel"]):
        # regular Monday grid: holiday gaps get linearly interpolated prices and a flag (D6)
        g = g.set_index("date").sort_index()
        grid = pd.date_range(g.index.min(), g.index.max(), freq="W-MON")
        g = g.reindex(grid).rename_axis("date")
        price_cols = ["price_with_tax_eur", "price_pre_tax_eur"]
        g["interpolated"] = g[price_cols].isna().any(axis=1)
        g[price_cols] = g[price_cols].interpolate(method="time")
        g = g.reset_index().assign(country=ctr, fuel=fuel)
        dates = pd.DatetimeIndex(g.date)
        cur = cfg["currencies"][ctr]

        rate = asof(fx[cur], dates)
        g["currency"] = cur
        g["lcu_per_eur"] = rate["value"].values
        g["fx_age_days"] = (dates - pd.DatetimeIndex(rate["source_date"])).days.values
        g["price_with_tax_lcu"] = g.price_with_tax_eur * g.lcu_per_eur
        g["price_pre_tax_lcu"] = g.price_pre_tax_eur * g.lcu_per_eur
        g["taxes_total_lcu"] = g.price_with_tax_lcu - g.price_pre_tax_lcu

        g["vat_pct_sheet"] = tax_on_dates(taxes["vat"], ctr, fuel, dates).values
        g["excise_lcu_sheet"] = tax_on_dates(taxes["excise"], ctr, fuel, dates).values / utils.LITRES_PER_KL
        g["other_taxes_lcu_sheet"] = tax_on_dates(taxes["other_taxes"], ctr, fuel, dates).values / utils.LITRES_PER_KL
        valid_vat = sorted(taxes["vat"].loc[taxes["vat"].country == ctr, "value"].unique())
        g = effective_taxes(g, valid_vat, cfg["build"])

        daily = brent_lcu_per_litre(brent, fx, cur)
        g["brent_usd_same_day"] = asof(brent, dates)["value"].values
        g["brent_lcu_same_day"] = asof(daily, dates)["value"].values
        week_before = [(d - pd.Timedelta(days=7), d) for d in dates]
        g["brent_usd_prev_week"] = [brent[(brent.index >= a) & (brent.index < b)].mean() for a, b in week_before]
        g["brent_lcu_prev_week"] = [daily[(daily.index >= a) & (daily.index < b)].mean() for a, b in week_before]

        sub = iv[(iv.country == ctr) & (iv.fuel == fuel)]
        g["intervention"] = [";".join(sub[(sub.start <= d) & (sub.end >= d)].measure) for d in dates]
        g["in_intervention"] = g.intervention != ""
        for ptype in cfg["policy_types"]:
            m = sub[sub.type == ptype]
            g[f"policy_{ptype}"] = [bool(((m.start <= d) & (m.end >= d)).any()) for d in dates]
        rows.append(g)
    return pd.concat(rows).reset_index(drop=True)


def build_hicp_monthly(snap, cfg, fx, brent, iv, weekly):
    hicp = load_hicp(snap, cfg)
    weights = load_weights(snap, cfg)
    hicp["year"] = hicp.month.dt.year
    hicp = hicp.merge(weights, on=["country", "coicop", "year"], how="left")

    def monthly_mean(s):
        return s.groupby(s.index.to_period("M")).mean().rename_axis("month").to_timestamp()

    drivers = []
    for ctr in cfg["countries"]:
        cur = cfg["currencies"][ctr]
        d = pd.DataFrame({
            "brent_usd": monthly_mean(brent),
            "usd_per_eur": monthly_mean(fx["USD"].dropna()),
            "lcu_per_eur": monthly_mean(fx[cur].dropna()),
            "brent_lcu": monthly_mean(brent_lcu_per_litre(brent, fx, cur)),
        })
        w = weekly[weekly.country == ctr].copy()
        w["month"] = w.date.dt.to_period("M").dt.to_timestamp()
        for fuel in cfg["series"]["oil_bulletin"]["fuels"]:
            wf = w[w.fuel == fuel].groupby("month")
            d[f"{fuel}_with_tax_lcu"] = wf.price_with_tax_lcu.mean()
            d[f"{fuel}_pre_tax_lcu"] = wf.price_pre_tax_lcu.mean()
        d["tax_change"] = w.groupby("month").tax_change.any().reindex(d.index, fill_value=False).astype(bool)

        # share of days in the month with at least one non-tax policy measure
        days = pd.date_range(d.index.min(), d.index.max() + pd.offsets.MonthEnd(0), freq="D")
        sub = iv[iv.country == ctr].drop_duplicates(["measure", "start", "end"])
        active = pd.Series(False, index=days)
        names = pd.Series("", index=days, dtype=object)
        for _, m in sub.iterrows():
            mask = (days >= m.start) & (days <= m.end)
            active |= mask
            names[mask] = names[mask].where(names[mask] == "", names[mask] + ";") + m.measure
        d["intervention_share"] = active.groupby(active.index.to_period("M")).mean().to_timestamp()
        for ptype in cfg["policy_types"]:
            on = pd.Series(False, index=days)
            for _, m in sub[sub.type == ptype].iterrows():
                on |= (days >= m.start) & (days <= m.end)
            d[f"share_{ptype}"] = on.groupby(on.index.to_period("M")).mean().to_timestamp()
        d["interventions"] = names.groupby(names.index.to_period("M")).agg(
            lambda x: ";".join(sorted({n for v in x for n in v.split(";") if n}))).to_timestamp()
        d["country"] = ctr
        drivers.append(d.rename_axis("month").reset_index())
    drivers = pd.concat(drivers)
    out = hicp.merge(drivers, on=["country", "month"], how="left").drop(columns="year")
    return out.sort_values(["country", "component", "month"]).reset_index(drop=True)


# --- Sanity checks ---------------------------------------------------------------------

def check(weekly, monthly, cfg):
    problems = []
    b = cfg["build"]

    if weekly.duplicated(["country", "fuel", "date"]).any():
        problems.append("duplicate country/fuel/date rows in fuel_weekly")
    if monthly.duplicated(["country", "component", "month"]).any():
        problems.append("duplicate country/component/month rows in hicp_monthly")

    bad = weekly[weekly.price_with_tax_lcu < weekly.price_pre_tax_lcu]
    if len(bad):
        problems.append(f"{len(bad)} weeks where the price with tax is below the price without tax")

    stale = weekly[weekly.fx_age_days > b["fx_max_staleness_days"]]
    if len(stale):
        problems.append(f"{len(stale)} bulletin dates without an ECB rate in the previous "
                        f"{b['fx_max_staleness_days']} days")

    for ctr, first in cfg["series"]["oil_bulletin"]["first_week"].items():
        actual = weekly.loc[weekly.country == ctr, "date"].min()
        if actual != pd.Timestamp(first):
            problems.append(f"{ctr}: first bulletin week is {actual.date()}, config says {first}")

    for ctr in cfg["countries"]:
        found = set(monthly.loc[monthly.country == ctr, "component"])
        missing = set(cfg["series"]["hicp_index"]["components"]) - found
        if missing:
            problems.append(f"{ctr}: HICP components missing: {sorted(missing)}")

    total = monthly[monthly.component == "headline"].drop_duplicates(["country", "weight"])
    total = total.dropna(subset=["weight"])
    if not np.allclose(total.weight, 1000):
        problems.append("headline HICP weights do not sum to 1000 in every year")

    # D4: the exchange rate implied by the bulletin's own taxes should equal the ECB rate
    w = weekly[(weekly.date >= b["implied_fx_check_from"])].dropna(
        subset=["vat_pct_sheet", "excise_lcu_sheet", "price_with_tax_eur", "price_pre_tax_eur"])
    for ctr, g in w.groupby("country"):
        taxes_lcu = g.excise_lcu_sheet + g.other_taxes_lcu_sheet.fillna(0)
        taxes_eur = g.price_with_tax_eur / (1 + g.vat_pct_sheet / 100) - g.price_pre_tax_eur
        diff_pct = ((taxes_lcu / taxes_eur) / g.lcu_per_eur - 1).abs().median() * 100
        if diff_pct > b["implied_fx_tolerance_pct"]:
            problems.append(f"{ctr}: bulletin's implied exchange rate differs from ECB by {diff_pct:.2f}% (median)")

    if problems:
        raise ValueError("Sanity checks failed:\n  - " + "\n  - ".join(problems))

    filled = weekly[weekly.interpolated].drop_duplicates(["country", "date"]).groupby("country").size()
    print(f"    checks passed; holiday weeks filled by interpolation (decisions D6): {filled.to_dict()}")


# --- Entry point (called by run.py) -------------------------------------------------------

def run(cfg):
    snap = utils.latest_snapshot()
    if snap is None:
        raise RuntimeError("No raw snapshot found - run `python run.py --refresh` first")
    print(f"    using snapshot {snap.name}")

    fx = load_fx(snap)
    brent = load_brent(snap, cfg)
    iv = load_interventions(cfg)

    weekly = build_fuel_weekly(snap, cfg, fx, brent, iv)
    monthly = build_hicp_monthly(snap, cfg, fx, brent, iv, weekly)
    check(weekly, monthly, cfg)
    report_tax_mismatches(weekly)

    utils.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_parquet(utils.DATA_PROCESSED / "fuel_weekly.parquet", index=False)
    monthly.to_parquet(utils.DATA_PROCESSED / "hicp_monthly.parquet", index=False)
    print(f"    fuel_weekly: {len(weekly):,} rows, {weekly.date.min().date()} to {weekly.date.max().date()}")
    print(f"    hicp_monthly: {len(monthly):,} rows, "
          f"{monthly.month.min():%Y-%m} to {monthly.month.max():%Y-%m}")