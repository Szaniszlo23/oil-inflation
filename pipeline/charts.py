"""
charts.py - figures and tables for the presentation, from saved results only.

Reads outputs/results/ and data/processed/ (plus interventions.csv for shading);
never estimates anything. Stops if any series in config.yaml is not verified.

Writes outputs/figures/:
  01_context            Brent and the contribution of fuels to HICP inflation
  02_pump_price         what makes up the petrol price: crude, margins, taxes
  03_crude_to_pump      Stage 1: cumulative pass-through, normal times
  04_policy_stage1      Stage 1: normal times vs. a measure in force vs. ignoring policy (04b: Hungary only)
  05_headline           Stage 2: headline HICP response
  06_components         Stage 2: component responses after 12 months
  07_direct_indirect    Stage 2: headline response split into contributions (fuels, other energy, food, core)
  08_chain              Brent -> pump price -> HICP fuels after 2 months
  09_counterfactual_2026          2026 interventions: actual vs. counterfactual pump prices
  10_counterfactual_hungary_2021  Hungary's 2021-22 cap and the post-cap premium
  11_counterfactual_inflation     direct effect of the interventions on headline inflation
  12_crosscheck_hungary_2021      Hungary's cap without a model: margin vs. Poland and Romania
  13_sample_comparison            Stage 2 responses before 2021 vs. the full sample (table)
  14_petrol_diesel                Stage 2: HICP petrol vs. diesel, with the Stage 1 long-run reference
  15_country_drivers              why the countries differ: structural drivers next to outcomes (table)
and outputs/tables/country_comparison.csv, policy_episodes.csv.

Stage 1 headline numbers (chart 03, comparison table) use the 2008-2019 sample, where the long-run
link between crude and pre-tax prices is clearest (cointegration tests: clear for PL and RO, weaker
for HU; stage1_diagnostics.csv); 2008-2026 enters through the model in differences, which does not
need a long-run link and gives the same 4-week pass-through.
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


def save(fig, name, note="", note_y=-0.02, rect=(0, 0, 1, 0.93)):
    snap = utils.latest_snapshot()
    if fig._suptitle is not None:
        fig.tight_layout(rect=rect)
    text = f"{SOURCE} Data as of {snap.name}." + (f" {note}" if note else "")
    fig.text(0.01, note_y, text, fontsize=7.5, color=COLORS["grey"], ha="left", va="top", wrap=True)
    fig.savefig(utils.FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"    {name}.png")


def note_below(fig, table):
    """Figure y-position just under a matplotlib table, for the footnote (avoids empty space)."""
    fig.canvas.draw()
    return table.get_window_extent().transformed(fig.transFigure.inverted()).y0 - 0.02


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


def s1_main(s1, version="interactions", sample="main"):
    return s1[(s1["sample"] == sample) & (s1.spec == "baseline") & (s1.policy_handling == version)]


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
    fig.suptitle("Petrol price before and after tax: what it is made of", x=0.01, ha="left", fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0))
    save(fig, "02_pump_price", "Red strip: price caps, maximum prices, margin caps, discounts and Hungary's post-cap window.",
         note_y=-0.01, rect=(0, 0.07, 1, 0.93))


def fig_crude_to_pump(cfg, s1, s1d):
    m = s1_main(s1, sample="pre2020")
    dl = s1d[(s1d.test == "dl_cumulative") & (s1d["sample"] == "main")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, ctr in zip(axes, cfg["countries"]):
        for fuel in cfg["stage1"]["fuels"]:
            d = m[(m.country == ctr) & (m.fuel == fuel) & (m.regime == "normal") & (m.measure == "cumulative")
                  & (m.horizon <= 12)].sort_values("horizon")
            ax.plot(d.horizon, d.estimate, FUEL_STYLE(fuel), color=C[ctr], lw=2, marker="o", ms=4,
                    label=f"{fuel.capitalize()}, 2008-19")
            ax.fill_between(d.horizon, d.lower, d.upper, color=C[ctr], alpha=0.12, lw=0)
            r = dl[(dl.country == ctr) & (dl.fuel == fuel) & (dl.horizon <= 12)].sort_values("horizon")
            ax.plot(r.horizon, r.statistic, ls="none", marker="x" if fuel == "petrol" else "+", ms=7,
                    color="black", label=f"{fuel.capitalize()}, 2008-26 (differences)")
        ax.axhline(1, color=COLORS["grey"], lw=0.8, ls=":")
        ax.set_xlim(-0.5, 12.5)
        ax.set_title(NAMES[ctr])
        ax.set_xlabel("weeks after the crude price move")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("pass-through (1 = one-for-one)")
    fig.suptitle("Pre-tax pump price response to a 1-unit move in crude (local currency per litre), normal times",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "03_crude_to_pump", "Lines: Stage 1 error-correction model, 2008-2019 (long-run link clear for Poland "
         "and Romania, weaker for Hungary), 95% bands. Markers: distributed lag in differences, 2008-2026 (needs no "
         "long-run link). Dotted line: one-for-one.")


def FUEL_STYLE(fuel):
    return utils.FUEL_STYLES[fuel]


def fig_policy_stage1(cfg, s1, countries=None, name="04_policy_stage1"):
    """Per country and fuel: normal times, the country's main type of measure in force, policy ignored.
    `countries` limits the chart (the Hungary-only version is the one precise enough for the main deck)."""
    subset = countries is not None
    countries = countries or cfg["countries"]
    m, ig = s1_main(s1), s1_main(s1, "ignore")
    kinds = [("Normal times", COLORS["crude"]), ("Measure in force", COLORS["accent"]),
             ("Policy ignored", "#A6A6A6")]
    lo, hi = (0.0, 1.25) if subset else cfg["charts"]["policy_chart_range"]

    def get(df, ctr, fuel, regime):
        d = df[(df.country == ctr) & (df.fuel == fuel) & (df.regime == regime)
               & (df.measure == "cumulative") & (df.horizon == 4)]
        return d.iloc[0] if len(d) else None

    labels, clipped = [], False
    fig, ax = plt.subplots(figsize=(10, 1.1 + 0.65 * len(countries) * len(cfg["stage1"]["fuels"])))
    j = 0
    for ctr in countries:
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
    if subset:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        ax.set_title(f"{', '.join(NAMES[c] for c in countries)}: crude to pre-tax pump price after 4 weeks, "
                     "normal times vs. under the price cap")
    else:
        ax.legend(loc="lower right")
        ax.set_title("Crude to pre-tax pump price after 4 weeks: normal times, with a measure in force, policy ignored")
    note = "Stage 1, 95% bands" + (" (clipped at the axis)" if clipped else "") + \
           ". 'Policy ignored': same model without any policy treatment."
    save(fig, name, note, note_y=-0.24 if subset else -0.02)


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
                    "administered": "Administered prices", "headline": "Headline"}


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


CONTRIB_PARTS = [("fuels", "Fuels (direct)", COLORS["crude"]),
                 ("other_energy", "Electricity, gas, heat (direct)", COLORS["margin"]),
                 ("food", "Food (indirect)", "#E46C0A"),
                 ("core", "Core (indirect)", "#A6A6A6")]


def fig_petrol_diesel(cfg, s1, s2):
    """HICP petrol vs. diesel response paths, with the Stage 1 implied long-run retail response as a reference."""
    m = s2_main(s2)
    el = s1_main(s1)
    el = el[(el.regime == "normal") & (el.measure == "retail_elasticity")]
    k = cfg["stage2"]["scale"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, ctr in zip(axes, cfg["countries"]):
        for fuel in cfg["stage1"]["fuels"]:
            d = m[(m.country == ctr) & (m.component == fuel) & (m.regime == "normal")
                  & (m.measure == "response")].sort_values("horizon")
            ax.plot(d.horizon, d.estimate, FUEL_STYLE(fuel), color=C[ctr], lw=2, label=f"HICP {fuel}")
            ax.fill_between(d.horizon, d.lower, d.upper, color=C[ctr], alpha=0.10, lw=0)
            ref = k * el[(el.country == ctr) & (el.fuel == fuel)].estimate.iloc[0]
            ax.axhline(ref, color=COLORS["grey"], lw=1, ls=FUEL_STYLE(fuel),
                       label=f"Stage 1 long-run, {fuel}")
        ax.axhline(0, color=COLORS["grey"], lw=0.8)
        ax.set_title(NAMES[ctr])
        ax.set_xlabel("months after the oil price rise")
        ax.legend(loc="lower right", fontsize=7.5)
    axes[0].set_ylabel("% change")
    fig.suptitle(f"Petrol and diesel in the HICP: response to a {k}% oil price rise, 2015-2026",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "14_petrol_diesel",
         "Stage 2 local projections on the HICP petrol and diesel indices (available from 2014-12), each with its own "
         "fuel's tax changes held constant, 90% bands. Grey lines: Stage 1 retail-price elasticity x "
         f"{k} (long-run pass-through to the price with tax, 2008-26, last two years' price structure).")


def fig_direct_indirect(cfg, s2):
    """Contributions to the headline response: 3 months, and 12 months before 2021 vs. the full sample."""
    c = s2[s2.spec == "contribution"]
    bars = [(3, "main", "3 months"), (12, "pre2021", "12 months\n2008-21"), (12, "main", "12 months\n2008-26")]
    width, gap = 0.8, 1.2
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ticks, labels = [], []
    for i, ctr in enumerate(cfg["countries"]):
        for j, (h, sample, lab) in enumerate(bars):
            x = i * (len(bars) + gap) + j
            d = c[(c.country == ctr) & (c["sample"] == sample) & (c.horizon == h)].set_index("measure")
            pos = neg = 0.0
            for part, name, col in CONTRIB_PARTS:
                v = d.loc[f"contrib_{part}", "estimate"]
                ax.bar(x, v, bottom=pos if v >= 0 else neg, color=col, width=width,
                       label=name if (i, j) == (0, 0) else None)
                pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
            t = d.loc["contrib_total"]
            ax.errorbar(x, t.estimate, yerr=[[t.estimate - t.lower], [t.upper - t.estimate]], fmt="D",
                        color="black", ms=5, capsize=3, lw=1, label="Total, 90% band" if (i, j) == (0, 0) else None)
            ticks.append(x)
            labels.append(lab)
        ax.text(i * (len(bars) + gap) + (len(bars) - 1) / 2, 1.02, NAMES[ctr], transform=ax.get_xaxis_transform(),
                ha="center", fontweight="bold", color=C[ctr])
    ax.set_xticks(ticks, labels, fontsize=8)
    ax.axhline(0, color=COLORS["grey"], lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("pp of headline HICP")
    ax.legend(loc="upper left", fontsize=8, ncol=1)
    fig.suptitle(f"What drives the headline response to a {cfg['stage2']['scale']}% oil price rise: "
                 "direct energy vs. indirect effects", x=0.01, ha="left", fontweight="bold")
    save(fig, "07_direct_indirect",
         "Stage 2 local projections on each component's contribution to headline inflation, year-specific HICP "
         "weights; parts add up to the total. Energy, food (incl. alcohol, tobacco) and core cover the whole basket.")


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
    method = ("Counterfactual: Stage 1 normal-times model run on the actual Brent path from the week before each "
              "measure. Shaded: measure in force.")
    note = method + (" Crude-only model: in 2026 the gap also contains changes in refining margins, and all three "
                     "countries intervened (no control group), so market effects are indicative.")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, ctr in zip(axes, cfg["countries"]):
        d = w[(w.country == ctr) & (w.episode_start >= recent)].sort_values("date")
        for i, (_, e) in enumerate(d.groupby("episode_start")):
            _cf_lines(ax, e, C[ctr], i == 0)
        ax.set_title(f"{NAMES[ctr]}, petrol, {cfg['currencies'][ctr]} per litre")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"{recent.year} interventions: actual pump prices vs. model counterfactual",
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
        save(fig, "10_counterfactual_hungary_2021", method + " Shading includes the post-cap window to Sep 2023. "
             "Crude-only model: it misses the 2022-23 rise in refining margins (especially diesel), so it overstates "
             "the post-cap premium; the model-free cross-check (chart 12) puts it at about 5-9%.")


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
         "Fuel weight x (actual - counterfactual annual change of the fuels index). Direct effect only. After a "
         "measure ends, annual inflation is compared with the capped prices a year earlier, so it turns positive: "
         "the cap shifts inflation into the following year rather than removing it (Hungary 2023).")


def fig_crosscheck(cfg, cc, pol, iv):
    """Hungary 2021-23 without a model: margin over crude vs. control countries, and the model for comparison."""
    c = cfg["policy"]["crosscheck"]
    cc = cc.assign(date=pd.to_datetime(cc.date)).set_index("date")
    mo = pol[(pol.level == "monthly") & (pol.country == c["country"])].assign(date=lambda d: pd.to_datetime(d.date))
    mo = mo.set_index("date").reindex(cc.index)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for a in (a1, a2):
        shade_policy(a, iv, c["country"], start=cc.index.min())
        a.axhline(0, color=COLORS["grey"], lw=0.8)
        a.set_xlim(cc.index.min(), cc.index.max() + pd.DateOffset(months=1))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    for fuel in cfg["stage1"]["fuels"]:
        a1.plot(cc.index, cc[f"effect_pct_{fuel}"], FUEL_STYLE(fuel), color=C[c["country"]], lw=2,
                label=f"{fuel.capitalize()}, vs. {' and '.join(NAMES[k] for k in c['controls'])}")
    a1.plot(cc.index, mo["price_level_gap_market_pct"], color="black", lw=1.2, ls=":",
            label="Model counterfactual (avg of fuels)")
    a1.set_title("Pump price vs. counterfactual, %")
    a1.legend(loc="lower left", fontsize=7.5)
    a2.plot(cc.index, cc["headline_effect_pp"], color=C[c["country"]], lw=2, label="Cross-check")
    a2.plot(cc.index, mo["headline_effect_market_pp"], color="black", lw=1.2, ls=":", label="Model counterfactual")
    a2.set_title("Direct effect on headline HICP inflation (y/y), pp")
    a2.legend(loc="lower left", fontsize=7.5)
    fig.suptitle(f"{NAMES[c['country']]}'s fuel price cap without a model: two methods, same story",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "12_crosscheck_hungary_2021",
         f"Cross-check: change in {NAMES[c['country']]}'s pre-tax margin over crude relative to the average of "
         f"{' and '.join(NAMES[k] for k in c['controls'])} (their weeks with own measures left out), baseline "
         f"{c['baseline'][0][:7]} to {c['baseline'][1][:7]}. Shaded: cap and post-cap window. Market measures only.")


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


def sample_table(cfg, s2, horizons=(3, 12)):
    """Stage 2 responses before 2021 vs. the full sample: how much of the effect comes from 2021-23.
    Writes a CSV and a slide-ready image; cells where the full-sample estimate lies outside the
    2008-21 band are highlighted."""
    comps = ["headline", "fuels", "energy", "electricity_gas", "transport_services", "food", "core"]
    samples = [("pre2021", "baseline", "2008-21"), ("main", "baseline", "2008-26"),
               ("main", "gas_control", "gas fixed"), ("main", "constant_taxes", "const. taxes")]
    r = s2[(s2.regime == "normal") & (s2.measure == "response")]
    r = r[r.policy_handling == r.country.map(                       # the main estimate's policy version
        s2[s2.main_estimate].drop_duplicates("country").set_index("country").policy_handling)]

    rows, cells, outside = [], [], []
    for comp in comps:
        for h in horizons:
            row, cell, out = {"component": COMPONENT_LABELS[comp], "months": h}, [], []
            for ctr in cfg["countries"]:
                est = {}
                for sample, spec, label in samples:
                    d = r[(r.country == ctr) & (r.component == comp) & (r["sample"] == sample)
                          & (r.spec == spec) & (r.horizon == h)]
                    est[(sample, spec)] = d.iloc[0]
                    row[f"{NAMES[ctr]} {label}"] = round(d.estimate.iloc[0], 2)
                    row[f"{NAMES[ctr]} {label} band"] = f"[{d.lower.iloc[0]:.2f}, {d.upper.iloc[0]:.2f}]"
                    cell.append(f"{d.estimate.iloc[0]:.2f}")
                pre, full = est[("pre2021", "baseline")], est[("main", "baseline")]
                out += [False, not pre.lower <= full.estimate <= pre.upper, False, False]
            rows.append(row)
            cells.append(cell)
            outside.append(out)
    pd.DataFrame(rows).to_csv(utils.TABLES / "sample_comparison.csv", index=False)
    print("    sample_comparison.csv")

    n = len(samples)
    fig, ax = plt.subplots(figsize=(15, 0.22 * len(cells) + 1.0))
    ax.axis("off")
    col_labels = [lab for _ in cfg["countries"] for _, _, lab in samples]
    row_labels = [f"{r['component']}, {r['months']}m" for r in rows]
    t = ax.table(cellText=cells, rowLabels=row_labels, colLabels=col_labels, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.35)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor(COLORS["light"])
        if i == 0 or j == -1:
            cell.set_text_props(fontweight="bold")
        if i > 0 and j >= 0 and outside[i - 1][j]:
            cell.set_facecolor(COLORS["policy"])
            cell.set_text_props(fontweight="bold")
    fig.canvas.draw()
    for k, ctr in enumerate(cfg["countries"]):         # country header above its group of columns
        a, b = t[0, n * k].get_window_extent(), t[0, n * k + n - 1].get_window_extent()
        x = ax.transAxes.inverted().transform(((a.x0 + b.x1) / 2, a.y1))
        ax.text(x[0], x[1] + 0.01, NAMES[ctr], transform=ax.transAxes, ha="center", va="bottom",
                fontweight="bold", color=C[ctr])
    ax.set_title(f"Response to a {cfg['stage2']['scale']}% oil price rise, %: before 2021, 2008-26 "
                 "(incl. 2021-23 and 2026), and two robustness checks on 2008-26", pad=28)
    save(fig, "13_sample_comparison",
         "Stage 2 local projections, % change of each HICP component. Shaded: the 2008-26 estimate lies outside "
         "the 90% band of the 2008-21 estimate. Gas fixed: EU gas price change (current and 3 lags) added; a lower "
         "bound, as gas was partly oil-indexed before ~2015. Const. taxes: HICP at constant tax rates.", note_y=note_below(fig, t))


def driver_table(cfg, weekly, monthly, s1, s2, recent_weeks=52):
    """Why the countries differ: structural drivers next to the outcomes they explain."""
    m2, m1 = s2_main(s2), s1_main(s1, sample="pre2020")
    contrib = s2[s2.spec == "contribution"]
    start = cfg["stage2"]["samples"]["main"]["start"]
    sections = {"Drivers": [], "Outcomes": []}
    values = {}
    for ctr in cfg["countries"]:
        mo = monthly[monthly.country == ctr]
        latest = mo[mo.month == mo.month.max()].set_index("component")["weight"]
        hist = mo[mo.month >= start]
        w = weekly[(weekly.country == ctr) & ~weekly.in_intervention]
        w = w[w.date >= w.date.max() - pd.Timedelta(weeks=recent_weeks)]
        resp = lambda comp, h, meas="response", src=m2: src[
            (src.country == ctr) & (src.component == comp) & (src.horizon == h) & (src.regime == "normal")
            & (src.measure == meas)].estimate.iloc[0]
        s1v = lambda meas, h=None: m1[(m1.country == ctr) & (m1.regime == "normal") & (m1.measure == meas)
                                      & ((m1.horizon == h) if h is not None else m1.horizon.isna())].estimate.mean()
        pre = s2[(s2["sample"] == "pre2021") & (s2.spec == "baseline") & (s2.regime == "normal")
                 & (s2.measure == "response") & (s2.country == ctr) & (s2.component == "headline")
                 & (s2.horizon == 12)].estimate.iloc[0]
        c12 = contrib[(contrib.country == ctr) & (contrib["sample"] == "main") & (contrib.horizon == 12)]
        c12 = c12.set_index("measure").estimate
        values[ctr] = {
            "Drivers": {
                "Fuel weight in HICP, per mille (2026 / avg 2008-26)":
                    f"{latest['fuels']:.0f} / {hist[hist.component == 'fuels'].drop_duplicates('month').weight.mean():.0f}",
                "Petrol share of fuel weight, % (2026)": f"{100 * latest['petrol'] / (latest['petrol'] + latest['diesel']):.0f}",
                "Crude share of pump price incl. VAT, % (last 52 wks)":
                    f"{100 * ((1 + w.vat_pct / 100) * w.brent_lcu_prev_week / w.price_with_tax_lcu).mean():.0f}",
                "Per-litre taxes (excise etc.), % of pump price": f"{100 * (w.fixed_taxes_lcu / w.price_with_tax_lcu).mean():.0f}",
                "VAT on fuel, %": f"{w.vat_pct.iloc[-1]:.0f}",
                "HICP fuels after 1 month, % per 10% weaker currency vs USD": f"{resp('fuels', 1, 'fx_response'):.1f}",
                "Food weight / administered-price weight, per mille (2026)":
                    f"{latest['food']:.0f} / {latest['administered']:.0f}",
                "Months with a cap, margin cap or discount, 2008-26":
                    f"{int((hist.drop_duplicates('month').intervention_share > 0).sum())}",
            },
            "Outcomes": {
                "Pre-tax pump price: weeks to 90% of long run (2008-19)": f"{s1v('weeks_to_90pct'):.1f}",
                "Pre-tax pass-through after 4 weeks (2008-19)": f"{s1v('cumulative', 4):.2f}",
                "HICP fuels after 2 months, % per 10% oil": f"{resp('fuels', 2):.1f}",
                "Headline after 3 months, % per 10% oil": f"{resp('headline', 3):.2f}",
                "Headline after 12 months, 2008-21 / 2008-26": f"{pre:.2f} / {resp('headline', 12):.2f}",
                "Of which direct energy / food / core, pp (12m, 2008-26)":
                    f"{c12['contrib_direct']:.2f} / {c12['contrib_food']:.2f} / {c12['contrib_core']:.2f}",
            },
        }
    rows = [(sec, label) for sec in sections for label in values[cfg["countries"][0]][sec]]
    table = pd.DataFrame({NAMES[c]: [values[c][s][l] for s, l in rows] for c in cfg["countries"]},
                         index=pd.MultiIndex.from_tuples(rows, names=["section", "item"]))
    table.to_csv(utils.TABLES / "country_drivers.csv")
    print("    country_drivers.csv")

    cells, labels, header_rows = [], [], []
    for sec in sections:
        header_rows.append(len(cells))
        cells.append([""] * len(cfg["countries"]))
        labels.append(sec)
        for label in values[cfg["countries"][0]][sec]:
            cells.append(list(table.loc[(sec, label)]))
            labels.append("  " + label)
    fig, ax = plt.subplots(figsize=(11, 0.22 * len(cells) + 1.0))
    ax.axis("off")
    t = ax.table(cellText=cells, rowLabels=labels, colLabels=[NAMES[c] for c in cfg["countries"]],
                 loc="upper center", cellLoc="center", rowLoc="left", colWidths=[0.16] * len(cfg["countries"]))
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.35)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor(COLORS["light"])
        if i == 0:
            cell.set_text_props(fontweight="bold", color=C[cfg["countries"][j]])
        if i > 0 and (i - 1) in header_rows:
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(fontweight="bold")
    ax.set_title("Why pass-through differs across Poland, Romania and Hungary", loc="left")
    save(fig, "15_country_drivers",
         "Pump-price structure: last 52 bulletin weeks without a non-tax measure, average of petrol and diesel. "
         "Currency: Stage 2 coefficient on the local currency per USD, holding oil in USD fixed; mechanical benchmark "
         "= crude share x the currency's own move after 1 month (about 13%), i.e. about 4%.", note_y=note_below(fig, t))


# --- Table ---------------------------------------------------------------------------------------

def country_table(cfg, weekly, monthly, s1, s2, s1d):
    m1, m1_main, m2 = s1_main(s1, sample="pre2020"), s1_main(s1), s2_main(s2)
    dl = s1d[(s1d.test == "dl_cumulative") & (s1d["sample"] == "main") & (s1d.horizon == 4)]
    rows = []
    for ctr in cfg["countries"]:
        w = weekly[(weekly.country == ctr) & (~weekly.in_intervention)].sort_values("date")
        recent = w[w.date >= w.date.max() - pd.Timedelta(weeks=52)]
        latest = monthly[(monthly.country == ctr) & (monthly.month == monthly.month.max())].set_index("component")

        def s1v(measure, h=None, fuel=None, src=m1):
            d = src[(src.country == ctr) & (src.regime == "normal") & (src.measure == measure)]
            d = d[d.horizon == h] if h is not None else d
            d = d[d.fuel == fuel] if fuel else d
            return d.estimate.mean()

        def elasticity_now():
            """Pre-2020 pass-through with today's price structure: retail elasticity is long-run pass-through
            x crude share of the retail price, so rescale the main-sample elasticity by the ratio of the betas."""
            vals = []
            for fuel in cfg["stage1"]["fuels"]:
                vals.append(s1v("retail_elasticity", fuel=fuel, src=m1_main)
                            * s1v("long_run", fuel=fuel) / s1v("long_run", fuel=fuel, src=m1_main))
            return np.mean(vals)

        def s2v(comp, h, measure="response"):
            d = m2 if measure == "response" else s2[(s2["sample"] == "main") & (s2.component == comp)]
            d = d[(d.country == ctr) & (d.component == comp) & (d.horizon == h) & (d.measure == measure)
                  & (d.regime == "normal")]
            return d.estimate.iloc[0] if len(d) else np.nan

        def contrib(part, h=12, sample="main"):
            d = s2[(s2.spec == "contribution") & (s2.country == ctr) & (s2["sample"] == sample)
                   & (s2.horizon == h) & (s2.measure == f"contrib_{part}")]
            return d.estimate.iloc[0]

        rows.append({
            "country": NAMES[ctr],
            "fuel weight in HICP, per mille (latest)": latest.loc["fuels", "weight"],
            "taxes, % of petrol price (last 52 weeks)": 100 * (recent[recent.fuel == "petrol"].eval(
                "(price_with_tax_lcu - price_pre_tax_lcu) / price_with_tax_lcu")).mean(),
            "long-run pre-tax pass-through, 2008-19 (avg of fuels)": s1v("long_run"),
            "pre-tax pass-through after 4 weeks, petrol, 2008-19": s1v("cumulative", 4, "petrol"),
            "pre-tax pass-through after 4 weeks, diesel, 2008-19": s1v("cumulative", 4, "diesel"),
            "pre-tax pass-through after 4 weeks, avg of fuels, 2008-26 (differences)":
                dl[dl.country == ctr].statistic.mean(),
            "weeks to 90% of long-run effect, 2008-19 (avg of fuels)": s1v("weeks_to_90pct"),
            "retail-price elasticity, 2008-19 pass-through at today's prices (avg of fuels)": elasticity_now(),
            "HICP fuels after 2 months, % per 10% oil": s2v("fuels", 2),
            "headline after 3 months, % per 10% oil": s2v("headline", 3),
            "headline after 12 months, % per 10% oil": s2v("headline", 12),
            "core after 12 months, % per 10% oil": s2v("core", 12),
            "indirect (food + core) share of headline effect at 12 months, %":
                100 * contrib("indirect") / contrib("total"),
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
    s1d = pd.read_csv(utils.RESULTS / "stage1_diagnostics.csv")
    iv = load_interventions(cfg)
    start = cfg["charts"]["context_start"]

    fig_context(cfg, monthly, start)
    fig_pump_price(cfg, weekly, iv, start)
    fig_crude_to_pump(cfg, s1, s1d)
    fig_policy_stage1(cfg, s1)
    fig_policy_stage1(cfg, s1, ["HU"], "04b_policy_stage1_hungary")
    fig_headline(cfg, s2)
    fig_components(cfg, s2)
    fig_direct_indirect(cfg, s2)
    fig_chain(cfg, s2)
    sample_table(cfg, s2)
    fig_petrol_diesel(cfg, s1, s2)
    driver_table(cfg, weekly, monthly, s1, s2)
    pol_path = utils.RESULTS / "policy_counterfactual.csv"
    if pol_path.exists():
        pol = pd.read_csv(pol_path)
        fig_counterfactual_prices(cfg, pol)
        fig_counterfactual_inflation(cfg, pol)
        policy_table(pol)
        cc_path = utils.RESULTS / "policy_crosscheck.csv"
        if cc_path.exists():
            fig_crosscheck(cfg, pd.read_csv(cc_path), pol, iv)
    else:
        print("    note: outputs/results/policy_counterfactual.csv not found - charts 09-11 skipped "
              "(run: python run.py --from policy)")
    print(country_table(cfg, weekly, monthly, s1, s2, s1d).to_string())