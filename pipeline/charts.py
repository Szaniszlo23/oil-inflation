"""
charts.py - figures and tables for the presentation, from saved results only.

Reads outputs/results/ and data/processed/ (plus interventions.csv for shading);
never estimates anything. Stops if any series in config.yaml is not verified.

Writes outputs/figures/:
  01_context            Brent and the contribution of fuels to HICP inflation
  02_pump_price         what makes up the petrol price: crude, margins, taxes
  03_crude_to_pump      Stage 1: cumulative pass-through, normal times
  04_policy_stage1      Stage 1: normal times vs. a measure in force vs. ignoring policy
  05_headline           Stage 2: headline HICP response
  06_components         Stage 2: component responses after 12 months
  07_direct_indirect    Stage 2: headline response split into direct and indirect
  08_chain              Brent -> pump price -> HICP fuels after 2 months
  09_counterfactual_2026          2026 interventions: actual vs. counterfactual pump prices
  10_counterfactual_hungary_2021  Hungary's 2021-22 cap and the post-cap premium
  11_counterfactual_inflation     direct effect of the interventions on headline inflation
and outputs/tables/country_comparison.csv, policy_episodes.csv.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import utils
from pipeline.build import load_interventions

SOURCE = "Source: Eurostat, ECB, FRED (EIA), EC Weekly Oil Bulletin; own calculations."
C, NAMES, COLORS = utils.COUNTRY_COLORS, utils.COUNTRY_NAMES, utils.COLORS


# --- Helpers ------------------------------------------------------------------------------

def check_verified(cfg):
    bad = [name for name, s in cfg["series"].items() if not s.get("verified")]
    if bad:
        raise ValueError(f"Series not verified in config.yaml: {bad} - charts are not drawn from unverified data")


def save(fig, name, note="", note_y=-0.02):
    snap = utils.latest_snapshot()
    if fig._suptitle is not None:
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    text = f"{SOURCE} Data as of {snap.name}." + (f" {note}" if note else "")
    fig.text(0.01, note_y, text, fontsize=7.5, color=COLORS["grey"], ha="left", va="top", wrap=True)
    fig.savefig(utils.FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"    {name}.png")


def shade_policy(ax, iv, ctr, fuel=None, start=None, strip=False):
    """Mark periods with a non-tax measure: full-height shading, or a strip at the top of the panel."""
    sub = iv[iv.country == ctr]
    if fuel:
        sub = sub[sub.fuel == fuel]
    first = True
    for _, m in sub.drop_duplicates(["measure"]).iterrows():
        if start is not None and m.end < start:
            continue
        kw = {"ymin": 0.96, "ymax": 1.0, "color": COLORS["accent"]} if strip else {"color": COLORS["policy"]}
        ax.axvspan(max(m.start, start) if start is not None else m.start, m.end, lw=0, zorder=3 if strip else 0,
                   label="Measure in force" if first else None, **kw)
        first = False


def s1_main(s1, version="interactions"):
    return s1[(s1["sample"] == "main") & (s1.spec == "baseline") & (s1.policy_handling == version)]


def s2_main(s2):
    return s2[(s2["sample"] == "main") & (s2.spec == "baseline") & (s2.main_estimate)]


# --- Figures ----------------------------------------------------------------------------------

def fig_context(cfg, monthly, start):
    h = monthly[monthly.component == "fuels"].copy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    b = h.drop_duplicates("month").set_index("month").loc[start:, "brent_usd"]
    a1.plot(b.index, b.values, color=COLORS["crude"], lw=2)
    a1.set_title("Brent crude, USD per barrel (monthly average)")
    for ctr in cfg["countries"]:
        x = h[h.country == ctr].set_index("month").sort_index()
        yoy = 100 * (x["index"] / x["index"].shift(12) - 1)
        contrib = (x["weight"] / 1000 * yoy).loc[start:]
        a2.plot(contrib.index, contrib.values, color=C[ctr], lw=2, label=NAMES[ctr])
    a2.axhline(0, color=COLORS["grey"], lw=0.8)
    a2.set_title("Contribution of fuels to HICP inflation, pp (approx.)")
    a2.legend(loc="upper left")
    for a in (a1, a2):
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save(fig, "01_context", "Contribution approximated as the year's HICP weight x annual change of the fuels index.")


def fig_pump_price(cfg, weekly, iv, start):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, ctr in zip(axes, cfg["countries"]):
        w = weekly[(weekly.country == ctr) & (weekly.fuel == "petrol")].set_index("date").sort_index().loc[start:]
        crude = w["brent_lcu_prev_week"]
        margin = w["price_pre_tax_lcu"] - crude
        taxes = w["price_with_tax_lcu"] - w["price_pre_tax_lcu"]
        shade_policy(ax, iv, ctr, "petrol", pd.Timestamp(start), strip=True)
        ax.stackplot(w.index, crude, margin, taxes, colors=[COLORS["crude"], COLORS["margin"], COLORS["taxes"]],
                     labels=["Crude oil", "Refining and retail margin", "Taxes"], lw=0)
        ax.set_title(f"{NAMES[ctr]}, {cfg['currencies'][ctr]} per litre")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(loc="upper left")
    fig.suptitle("Petrol price before and after tax: what it is made of", x=0.01, ha="left", fontweight="bold")
    save(fig, "02_pump_price", "Red strip: price caps, maximum prices, margin caps, discounts and Hungary's post-cap window.")


def fig_crude_to_pump(cfg, s1):
    m = s1_main(s1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, ctr in zip(axes, cfg["countries"]):
        for fuel in cfg["stage1"]["fuels"]:
            d = m[(m.country == ctr) & (m.fuel == fuel) & (m.regime == "normal") & (m.measure == "cumulative")
                  & (m.horizon <= 12)].sort_values("horizon")
            ax.plot(d.horizon, d.estimate, FUEL_STYLE(fuel), color=C[ctr], lw=2, marker="o", ms=4,
                    label=fuel.capitalize())
            ax.fill_between(d.horizon, d.lower, d.upper, color=C[ctr], alpha=0.12, lw=0)
        ax.axhline(1, color=COLORS["grey"], lw=0.8, ls=":")
        ax.set_title(NAMES[ctr])
        ax.set_xlabel("weeks after the crude price move")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("pass-through (1 = one-for-one)")
    fig.suptitle("Pre-tax pump price response to a 1-unit move in crude (local currency per litre), normal times",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "03_crude_to_pump", "Stage 1 error-correction model, 2008-2026, 95% bands. Dotted line: one-for-one.")


def FUEL_STYLE(fuel):
    return utils.FUEL_STYLES[fuel]


def fig_policy_stage1(cfg, s1):
    """Per country and fuel: normal times, the country's main type of measure in force, policy ignored."""
    m, ig = s1_main(s1), s1_main(s1, "ignore")
    kinds = [("Normal times", COLORS["crude"]), ("Measure in force", COLORS["accent"]),
             ("Policy ignored", "#A6A6A6")]
    lo, hi = cfg["charts"]["policy_chart_range"]

    def get(df, ctr, fuel, regime):
        d = df[(df.country == ctr) & (df.fuel == fuel) & (df.regime == regime)
               & (df.measure == "cumulative") & (df.horizon == 4)]
        return d.iloc[0] if len(d) else None

    labels, clipped = [], False
    fig, ax = plt.subplots(figsize=(10, 5))
    j = 0
    for ctr in cfg["countries"]:
        ptype = cfg["charts"]["main_policy_type"][ctr]
        for fuel in cfg["stage1"]["fuels"]:
            values = [get(m, ctr, fuel, "normal"), get(m, ctr, fuel, ptype), get(ig, ctr, fuel, "normal")]
            for i, ((kind, col), r) in enumerate(zip(kinds, values)):
                if r is None:
                    continue
                y = j + (i - 1) * 0.27
                ax.barh(y, r.estimate, height=0.25, color=col, label=kind if j == 0 else None)
                lower, upper = max(r.lower, lo), min(r.upper, hi)
                clipped |= (r.lower < lo) or (r.upper > hi)
                ax.errorbar(r.estimate, y, xerr=[[r.estimate - lower], [upper - r.estimate]],
                            color="black", lw=0.8, capsize=2)
            labels.append(f"{NAMES[ctr]}, {fuel}\n({ptype.replace('_', ' ')})")
            j += 1
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(lo, hi)
    ax.axvline(1, color=COLORS["grey"], lw=0.8, ls=":")
    ax.grid(axis="x"), ax.grid(axis="y", visible=False)
    ax.set_xlabel("cumulative pass-through after 4 weeks (1 = one-for-one)")
    ax.legend(loc="lower right")
    ax.set_title("Crude to pre-tax pump price after 4 weeks: normal times, with a measure in force, policy ignored")
    note = "Stage 1, 95% bands" + (" (clipped at the axis)" if clipped else "") + \
           ". 'Policy ignored': same model without any policy treatment."
    save(fig, "04_policy_stage1", note)


def fig_headline(cfg, s2):
    m = s2_main(s2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, ctr in zip(axes, cfg["countries"]):
        d = m[(m.country == ctr) & (m.component == "headline") & (m.regime == "normal")
              & (m.measure == "response")].sort_values("horizon")
        ax.plot(d.horizon, d.estimate, color=C[ctr], lw=2.2)
        ax.fill_between(d.horizon, d.lower, d.upper, color=C[ctr], alpha=0.15, lw=0)
        ax.axhline(0, color=COLORS["grey"], lw=0.8)
        ax.set_title(NAMES[ctr])
        ax.set_xlabel("months after the oil price rise")
    axes[0].set_ylabel("% change in the HICP")
    fig.suptitle(f"Headline HICP response to a {cfg['stage2']['scale']}% rise in oil prices",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "05_headline", "Stage 2 local projections, 2008-2026, 90% bands; exchange rate and fuel tax changes held constant.")


COMPONENT_LABELS = {"fuels": "Fuels", "energy": "Energy", "electricity_gas": "Electricity, gas, heat",
                    "transport_services": "Transport services", "food": "Food", "core": "Core",
                    "headline": "Headline"}


def fig_components(cfg, s2, h=12):
    m = s2_main(s2)
    comps = list(COMPONENT_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for ax, ctr in zip(axes, cfg["countries"]):
        for i, comp in enumerate(comps):
            d = m[(m.country == ctr) & (m.component == comp) & (m.regime == "normal")
                  & (m.measure == "response") & (m.horizon == h)]
            if d.empty:
                continue
            r = d.iloc[0]
            col = C[ctr] if comp == "headline" else COLORS["grey"]
            ax.errorbar(r.estimate, i, xerr=[[r.estimate - r.lower], [r.upper - r.estimate]], fmt="o",
                        color=col, ms=6, capsize=3, lw=1.2)
        ax.set_yticks(range(len(comps)), [COMPONENT_LABELS[c] for c in comps] if ctr == cfg["countries"][0] else [])
        ax.invert_yaxis()
        ax.axvline(0, color=COLORS["grey"], lw=0.8)
        ax.grid(axis="x"), ax.grid(axis="y", visible=False)
        ax.set_title(NAMES[ctr])
        ax.set_xlabel("% change after 12 months")
    fig.suptitle(f"Which prices move: response to a {cfg['stage2']['scale']}% oil price rise after {h} months",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "06_components", "Stage 2 local projections, 2008-2026, 90% bands.")


def fig_direct_indirect(cfg, s2, h=12):
    m = s2[(s2["sample"] == "main") & (s2.component == "headline") & (s2.horizon == h)]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, ctr in enumerate(cfg["countries"]):
        get = lambda meas: m[(m.country == ctr) & (m.measure == meas)].estimate.iloc[0]
        fuels, energy, indirect = get("fuels_direct"), get("energy_direct"), get("indirect")
        parts = [(fuels, "Fuels (direct)", COLORS["crude"]),
                 (energy - fuels, "Other energy (direct)", COLORS["margin"]),
                 (indirect, "Indirect (other prices)", "#E46C0A")]
        pos, neg = 0.0, 0.0
        for v, lab, col in parts:
            base = pos if v >= 0 else neg
            ax.bar(i, v, bottom=base, color=col, width=0.55, label=lab if i == 0 else None)
            if v >= 0:
                pos += v
            else:
                neg += v
        total = m[(m.country == ctr) & (m.measure == "response") & m.main_estimate].estimate.iloc[0]
        ax.plot(i, total, marker="D", color="black", ms=6, label="Headline total" if i == 0 else None)
    ax.set_xticks(range(len(cfg["countries"])), [NAMES[c] for c in cfg["countries"]])
    ax.axhline(0, color=COLORS["grey"], lw=0.8)
    ax.set_ylabel("pp of headline HICP")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Headline response after {h} months to a {cfg['stage2']['scale']}% oil rise: direct vs. indirect")
    save(fig, "07_direct_indirect", "Direct = latest HICP weight x component response; indirect = headline minus energy.")


def fig_chain(cfg, s2, h=2):
    ch = s2[(s2.spec == "chain")]
    if not (ch.measure == "crude_share").any():
        raise RuntimeError("stage2_responses.csv has no pass-through chain results - it was produced by an older "
                           "stage2.py. Rerun: python run.py --from stage2")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = [("Brent in local currency", "brent_lcu", COLORS["crude"]),
            ("Full pass-through benchmark", None, COLORS["light"]),
            ("Pump price with tax", "pump_price", COLORS["margin"]),
            ("HICP fuels", "fuels", "#E46C0A")]
    width = 0.2
    for i, ctr in enumerate(cfg["countries"]):
        resp = lambda comp: ch[(ch.country == ctr) & (ch.component == comp) & (ch.measure == "response")
                               & (ch.horizon == h)].estimate.iloc[0]
        share = ch[(ch.country == ctr) & (ch.measure == "crude_share")].estimate.iloc[0]
        fuels = s2_main(s2)
        fuels = fuels[(fuels.country == ctr) & (fuels.component == "fuels") & (fuels.regime == "normal")
                      & (fuels.measure == "response") & (fuels.horizon == h)].estimate.iloc[0]
        values = [resp("brent_lcu"), share * resp("brent_lcu"), resp("pump_price"), fuels]
        for j, ((lab, _, col), v) in enumerate(zip(bars, values)):
            ax.bar(i + (j - 1.5) * width, v, width=width * 0.95, color=col, label=lab if i == 0 else None)
    ax.set_xticks(range(len(cfg["countries"])), [NAMES[c] for c in cfg["countries"]])
    ax.set_ylabel("% change")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8)
    ax.set_title(f"From crude to the HICP: responses {h} months after a {cfg['stage2']['scale']}% oil price rise")
    save(fig, "08_chain", "Benchmark = crude share of the retail price (VAT included) x move in Brent in local currency.",
         note_y=-0.1)


def _cf_lines(ax, e, color, first):
    on = e[e.in_force]
    ax.axvspan(on.date.min(), on.date.max(), color=COLORS["policy"], lw=0, zorder=0)
    ax.plot(e.date, e.retail_actual, color=color, lw=2, label="Actual" if first else None)
    ax.plot(e.date, e.retail_cf_market, color=COLORS["grey"], lw=1.5, ls="--",
            label="Without the market measure" if first else None)
    ax.plot(e.date, e.retail_cf_full, color="black", lw=1, ls=":",
            label="Without any measure (taxes as before)" if first else None)


def fig_counterfactual_prices(cfg, pol):
    """2026 episodes in all three countries, and Hungary's 2021-23 cap with its aftermath."""
    w = pol[(pol.level == "weekly") & (pol.fuel == "petrol")].copy()
    w["date"] = pd.to_datetime(w["date"])
    w["episode_start"] = pd.to_datetime(w["episode_start"])
    recent = pd.Timestamp(cfg["charts"]["recent_episodes_start"])
    note = ("Counterfactual: Stage 1 normal-times model run on the actual Brent path from the week before each "
            "measure. Shaded: measure in force.")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, ctr in zip(axes, cfg["countries"]):
        d = w[(w.country == ctr) & (w.episode_start >= recent)].sort_values("date")
        for i, (_, e) in enumerate(d.groupby("episode_start")):
            _cf_lines(ax, e, C[ctr], i == 0)
        ax.set_title(f"{NAMES[ctr]}, petrol, {cfg['currencies'][ctr]} per litre")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"2026 interventions: actual pump prices vs. model counterfactual ({recent.year})",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "09_counterfactual_2026", note)

    first_hu = cfg["charts"]["hungary_episode_start"]
    d = w[(w.country == "HU") & (w.episode_start == pd.Timestamp(first_hu))].sort_values("date")
    if len(d):
        fig, ax = plt.subplots(figsize=(9, 4))
        _cf_lines(ax, d, C["HU"], True)
        cap_end = d[d.phase.str.startswith("retail_price_cap")].date.max()
        ax.axvline(cap_end, color=COLORS["accent"], lw=1, ls="--")
        ax.text(cap_end, ax.get_ylim()[1], " cap lifted", color=COLORS["accent"], va="top", fontsize=8)
        ax.set_title("Hungary 2021-23: petrol price under the cap and after it, HUF per litre")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.legend(loc="upper left", fontsize=8)
        save(fig, "10_counterfactual_hungary_2021", note + " Shading includes the post-cap window to Sep 2023.")


def fig_counterfactual_inflation(cfg, pol):
    m = pol[pol.level == "monthly"].copy()
    m["date"] = pd.to_datetime(m["date"])
    start = pd.Timestamp(cfg["charts"]["policy_start"])
    end = m["date"].max()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True, sharex=True)
    for ax, ctr in zip(axes, cfg["countries"]):
        d = m[(m.country == ctr) & (m.date >= start)].sort_values("date").set_index("date")
        ax.set_xlim(start, end + pd.DateOffset(months=1))
        market = d["headline_effect_market_pp"].fillna(0)
        taxes = (d["headline_effect_full_pp"] - d["headline_effect_market_pp"]).fillna(0)
        width = 25
        ax.bar(d.index, market, width=width, color=C[ctr], label="Caps, margin caps, discounts")
        ax.bar(d.index, taxes, width=width, bottom=np.where((taxes >= 0) == (market >= 0), market, 0),
               color=COLORS["light"], label="Tax changes")
        ax.axhline(0, color=COLORS["grey"], lw=0.8)
        ax.set_title(NAMES[ctr])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper left", fontsize=7.5)
    axes[0].set_ylabel("pp of headline HICP inflation (y/y)")
    fig.suptitle("Direct effect of fuel interventions on headline inflation (y/y): negative = inflation lowered",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "11_counterfactual_inflation",
         "Fuel weight x (actual - counterfactual annual change of the fuels index). Direct effect only; "
         "includes base effects after a measure ends.")


def policy_table(pol):
    w = pol[pol.level == "weekly"]
    rows = []
    for (ctr, phase), d in w.groupby(["country", "phase"], sort=False):
        by_week = d.groupby("date")[["market_effect_pct", "tax_effect_pct", "total_effect_pct"]].mean()
        rows.append({"country": NAMES[ctr], "measure (phase)": phase,
                     "from": pd.to_datetime(by_week.index.min()).date(), "to": pd.to_datetime(by_week.index.max()).date(),
                     "weeks": len(by_week),
                     "pump price vs. counterfactual, market measures, avg %": by_week.market_effect_pct.mean(),
                     "... peak %": by_week.market_effect_pct.loc[by_week.market_effect_pct.abs().idxmax()],
                     "tax changes, avg %": by_week.tax_effect_pct.mean(),
                     "total, avg %": by_week.total_effect_pct.mean()})
    table = pd.DataFrame(rows).round(1)
    table.to_csv(utils.TABLES / "policy_episodes.csv", index=False)
    print("    policy_episodes.csv")


# --- Table ---------------------------------------------------------------------------------------

def country_table(cfg, weekly, monthly, s1, s2):
    m1, m2 = s1_main(s1), s2_main(s2)
    rows = []
    for ctr in cfg["countries"]:
        w = weekly[(weekly.country == ctr) & (~weekly.in_intervention)].sort_values("date")
        recent = w[w.date >= w.date.max() - pd.Timedelta(weeks=52)]
        latest = monthly[(monthly.country == ctr) & (monthly.month == monthly.month.max())].set_index("component")

        def s1v(measure, h=None, fuel=None):
            d = m1[(m1.country == ctr) & (m1.regime == "normal") & (m1.measure == measure)]
            d = d[d.horizon == h] if h is not None else d
            d = d[d.fuel == fuel] if fuel else d
            return d.estimate.mean()

        def s2v(comp, h, measure="response"):
            d = m2 if measure == "response" else s2[(s2["sample"] == "main") & (s2.component == comp)]
            d = d[(d.country == ctr) & (d.component == comp) & (d.horizon == h) & (d.measure == measure)
                  & (d.regime == "normal")]
            return d.estimate.iloc[0] if len(d) else np.nan

        rows.append({
            "country": NAMES[ctr],
            "fuel weight in HICP, per mille (latest)": latest.loc["fuels", "weight"],
            "taxes, % of petrol price (last 52 weeks)": 100 * (recent[recent.fuel == "petrol"].eval(
                "(price_with_tax_lcu - price_pre_tax_lcu) / price_with_tax_lcu")).mean(),
            "pre-tax pass-through after 4 weeks, petrol": s1v("cumulative", 4, "petrol"),
            "pre-tax pass-through after 4 weeks, diesel": s1v("cumulative", 4, "diesel"),
            "weeks to 50% of long-run effect (avg of fuels)": s1v("weeks_to_50pct"),
            "retail-price elasticity (avg of fuels)": s1v("retail_elasticity"),
            "HICP fuels after 2 months, % per 10% oil": s2v("fuels", 2),
            "headline after 3 months, % per 10% oil": s2v("headline", 3),
            "headline after 12 months, % per 10% oil": s2v("headline", 12),
            "core after 12 months, % per 10% oil": s2v("core", 12),
            "indirect share of headline effect at 12 months, %":
                100 * s2v("headline", 12, "indirect") / s2v("headline", 12),
        })
    table = pd.DataFrame(rows).set_index("country").T.round(2)
    table.to_csv(utils.TABLES / "country_comparison.csv")
    print("    country_comparison.csv")
    return table


# --- Entry point (called by run.py) -----------------------------------------------------------------

def run(cfg):
    check_verified(cfg)
    utils.apply_chart_style()
    utils.FIGURES.mkdir(parents=True, exist_ok=True)
    utils.TABLES.mkdir(parents=True, exist_ok=True)
    weekly = pd.read_parquet(utils.DATA_PROCESSED / "fuel_weekly.parquet")
    monthly = pd.read_parquet(utils.DATA_PROCESSED / "hicp_monthly.parquet")
    s1 = pd.read_csv(utils.RESULTS / "stage1_passthrough.csv")
    s2 = pd.read_csv(utils.RESULTS / "stage2_responses.csv")
    iv = load_interventions(cfg)
    start = cfg["charts"]["context_start"]

    fig_context(cfg, monthly, start)
    fig_pump_price(cfg, weekly, iv, start)
    fig_crude_to_pump(cfg, s1)
    fig_policy_stage1(cfg, s1)
    fig_headline(cfg, s2)
    fig_components(cfg, s2)
    fig_direct_indirect(cfg, s2)
    fig_chain(cfg, s2)
    pol_path = utils.RESULTS / "policy_counterfactual.csv"
    if pol_path.exists():
        pol = pd.read_csv(pol_path)
        fig_counterfactual_prices(cfg, pol)
        fig_counterfactual_inflation(cfg, pol)
        policy_table(pol)
    else:
        print("    note: outputs/results/policy_counterfactual.csv not found - charts 09-11 skipped "
              "(run: python run.py --from policy)")
    print(country_table(cfg, weekly, monthly, s1, s2).to_string())