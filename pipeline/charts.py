"""
charts.py - figures and tables for the presentation, from saved results only.

Reads outputs/results/ and data/processed/ (plus interventions.csv for shading);
never estimates anything. Stops if any series in config.yaml is not verified.

Writes outputs/figures/:
  01_context            Brent and the contribution of fuels to HICP inflation
  02_pump_price         what makes up the petrol price: crude, margins, taxes
  03_crude_to_pump      Stage 1, short run: cumulative pass-through, 2008-19 (03b: with the model in differences)
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
  16_energy_contributions         petrol, diesel, household gas, electricity, heating: contributions to headline
  17_chain_normal_times           Stage 1 and Stage 2 combined: crude -> pump price -> HICP fuels -> headline
  17b_chain_full_period           the same on 2008-26
  18_price_structure              pump price = 100%: crude, margin, excise, VAT, petrol and diesel, 2008-26 (table)
  19_crude_share_petrol           share of crude oil in the petrol price by year (table)
  20_long_run                     Stage 1, long run: pass-through 2008-19 vs 2008-26 by country and fuel
  21_long_run                     the same as a table with 95% confidence intervals
  22_pump_to_headline             pump prices to headline: worked example and Stage 1 vs Stage 2 comparison
  23_components                   headline response by component, 3 vs 12 months, 2008-21
  23b_components_by_month         the same month by month (backup)
  24_intervention_timeline        all fuel interventions since 2021: effect on headline, one line per country
  24b_intervention_bars           the same as bars, caps vs tax changes (backup)
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

SOURCE = "Source: Eurostat, ECB, FRED, EC Oil Bulletin; own calculations."
C, NAMES, COLORS = utils.COUNTRY_COLORS, utils.COUNTRY_NAMES, utils.COLORS


# --- Helpers ------------------------------------------------------------------------------

def check_verified(cfg):
    bad = [name for name, s in cfg["series"].items() if not s.get("verified")]
    if bad:
        raise ValueError(f"Series not verified in config.yaml: {bad} - charts are not drawn from unverified data")


def save(fig, name, note="", note_y=-0.02, rect=(0, 0, 1, 0.93), note_x=0.01):
    snap = utils.latest_snapshot()
    if fig._suptitle is not None:
        fig.tight_layout(rect=rect)
    text = SOURCE + (f" {note}" if note else "")
    fig.text(note_x, note_y, text, fontsize=7.5, color=COLORS["grey"], ha="left", va="top", wrap=True)
    fig.savefig(utils.FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"    {name}.png")


def note_below(fig, table):
    """Figure y-position just under a matplotlib table, for the footnote (avoids empty space)."""
    fig.canvas.draw()
    return table.get_window_extent().transformed(fig.transFigure.inverted()).y0 - 0.02


def table_left(fig, table):
    """Figure x-position of a matplotlib table's left edge (row labels included), to align the footnote."""
    fig.canvas.draw()
    return table.get_window_extent().transformed(fig.transFigure.inverted()).x0


def table_title(fig, table, text, gap_pt=10):
    """Title aligned with the table's left edge, gap_pt points above it (more if headers sit above the table)."""
    from matplotlib.transforms import ScaledTranslation
    fig.canvas.draw()
    bb = table.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(bb.x0, bb.y1, text, fontsize=11, fontweight="bold", ha="left", va="bottom",
             transform=fig.transFigure + ScaledTranslation(0, gap_pt / 72, fig.dpi_scale_trans))


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
    save(fig, "01_context", "Contribution = HICP weight x annual change of the fuels index.")


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
    save(fig, "02_pump_price", "Red strip: price measures in force.",
         note_y=-0.01, rect=(0, 0.07, 1, 0.93))


def fig_crude_to_pump(cfg, s1, s1d, markers=False, name="03_crude_to_pump"):
    """Short-run adjustment: cumulative pre-tax pass-through over 0-12 weeks, 2008-19 (step 2 of the
    error-correction model). markers=True adds the distributed lag in differences, 2008-26 (backup version)."""
    m = s1_main(s1, sample="pre2020")
    dl = s1d[(s1d.test == "dl_cumulative") & (s1d["sample"] == "main")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, ctr in zip(axes, cfg["countries"]):
        for fuel in cfg["stage1"]["fuels"]:
            d = m[(m.country == ctr) & (m.fuel == fuel) & (m.regime == "normal") & (m.measure == "cumulative")
                  & (m.horizon <= 12)].sort_values("horizon")
            ax.plot(d.horizon, d.estimate, FUEL_STYLE(fuel), color=C[ctr], lw=2, marker="o", ms=4,
                    label=fuel.capitalize() + (", 2008-19" if markers else ""))
            ax.fill_between(d.horizon, d.lower, d.upper, color=C[ctr], alpha=0.12, lw=0)
            if markers:
                r = dl[(dl.country == ctr) & (dl.fuel == fuel) & (dl.horizon <= 12)].sort_values("horizon")
                ax.plot(r.horizon, r.statistic, ls="none", marker="x" if fuel == "petrol" else "+", ms=7,
                        color="black", label=f"{fuel.capitalize()}, 2008-26 (differences)")
        ax.axhline(1, color=COLORS["grey"], lw=0.8, ls=":")
        ax.set_xlim(-0.5, 12.5)
        ax.set_xticks([0, 1, 2, 4, 8, 12])
        ax.set_title(NAMES[ctr])
        ax.set_xlabel("weeks after the crude price move")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("pass-through (1 = one-for-one)")
    fig.suptitle("Short-run adjustment: pre-tax pump price response to a 1-unit move in crude, 2008-19",
                 x=0.01, ha="left", fontweight="bold")
    note = "Stage 1 error-correction model; 95% bands."
    if markers:
        note += " Markers: model in differences, 2008-26."
    save(fig, name, note)


def fig_long_run(cfg, s1):
    """Long-run pass-through (step 1 of the error-correction model): 2008-19 vs 2008-26, by country and fuel."""
    rows = [(ctr, fuel) for ctr in cfg["countries"] for fuel in cfg["stage1"]["fuels"]]
    samples = [("pre2020", "2008-19", True), ("main", "2008-26", False)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y, labels = [], []
    for i, (ctr, fuel) in enumerate(rows):
        pos = i + (i // len(cfg["stage1"]["fuels"])) * 0.6           # gap between countries
        y.append(pos)
        labels.append(f"{NAMES[ctr]}, {fuel}")
        for k, (sample, lab, filled) in enumerate(samples):
            d = s1_main(s1, sample=sample)
            r = d[(d.country == ctr) & (d.fuel == fuel) & (d.regime == "normal") & (d.measure == "long_run")].iloc[0]
            yy = pos + (k - 0.5) * 0.3
            ax.errorbar(r.estimate, yy, xerr=[[r.estimate - r.lower], [r.upper - r.estimate]], fmt="o",
                        color=C[ctr], mfc=C[ctr] if filled else "white", ms=7, capsize=3, lw=1.2,
                        label=lab if i == 0 else None)
    ax.axvline(1, color=COLORS["grey"], lw=0.8, ls=":")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="x"), ax.grid(axis="y", visible=False)
    ax.set_xlabel("long-run pass-through (1 = one-for-one)")
    leg = ax.legend(loc="lower right")
    for h in leg.legend_handles:
        h.set_color(COLORS["grey"])
    ax.set_title("Long-run pass-through from crude to pre-tax pump prices", loc="left")
    save(fig, "20_long_run", "Dynamic OLS, weeks without price measures; 95% bands. Filled: 2008-19; hollow: 2008-26.")


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
    note = "Stage 1; 95% bands" + (" (clipped)" if clipped else "") + ". 'Policy ignored': no policy treatment."
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
    save(fig, "05_headline", "Local projections, 2008-26; 90% bands.")


COMPONENT_LABELS = {"fuels": "Fuels", "energy": "Energy", "electricity_gas": "Electricity, gas, heat",
                    "electricity": "  of which electricity", "household_gas": "  of which household gas",
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
    save(fig, "06_components", "Local projections, 2008-26; 90% bands.")


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
         "Local projections, 2015-26; 90% bands. Grey lines: Stage 1 long-run response of the pump price.")


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
         "Contributions with year-specific HICP weights; parts add up to the total.")


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
    save(fig, "08_chain", "Benchmark = crude share of the pump price x move in Brent (local currency).",
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
    method = "Counterfactual: Stage 1 model on the actual Brent path. Shaded: measure in force."
    note = method + " Indicative: all three countries intervened in 2026."

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
        save(fig, "10_counterfactual_hungary_2021", method + " The model overstates the post-cap premium (cross-check: ~5-9%).")


def fig_intervention_lines(cfg, tl):
    """All fuel interventions since 2021: total direct effect on headline inflation, one line per country,
    with a strip below showing when each measure was in force."""
    from matplotlib.patches import Patch
    tl = tl.assign(date=pd.to_datetime(tl.date))
    x0, x1 = pd.Timestamp("2020-11-01"), tl.date.max() + pd.DateOffset(months=2)
    fig, (ax, st) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [3.4, 1.15]})
    ax.axhspan(0, 10, color="#FBEAEA", lw=0, zorder=0)
    ax.axhspan(-10, 0, color="#EAF1F8", lw=0, zorder=0)
    periods = cfg["charts"].get("timeline_periods", [])
    for ctr in cfg["countries"]:
        d = tl[tl.country == ctr].set_index("date").sort_index().total_pp
        ax.plot(d.index, d.values, color=C[ctr], lw=2.4, label=NAMES[ctr], zorder=3)
    ax.axhline(0, color=COLORS["grey"], lw=1, zorder=2)
    lo, hi = tl.total_pp.min(), tl.total_pp.max()
    ax.set_ylim(lo - 0.5, hi + 0.5)
    ax.text(0.995, 0.97, "inflation raised", transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#9B2C2C")
    ax.text(0.995, 0.03, "inflation lowered", transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color="#1F4E79")
    for ctr, month, text, tx, ty in cfg["charts"].get("timeline_line_labels", []):
        d = tl[tl.country == ctr].set_index("date").total_pp
        t = pd.Timestamp(month)
        ax.annotate(text, xy=(t, d.get(t, 0.0)), xytext=(pd.Timestamp(tx), ty), fontsize=9.5, color="#1A1A1A",
                    va="center", arrowprops=dict(arrowstyle="-", color=COLORS["grey"], lw=0.8), zorder=4)
    ax.set_ylabel("pp of headline inflation (y/y)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3, fontsize=9.5)
    ax.set_title("Direct effect of fuel price caps and tax changes on headline inflation", loc="left")

    # strip: one row per country, a bar for each period with a measure in force
    rows = {c: i for i, c in enumerate(cfg["countries"])}
    for ctr, a, b, label, kind in periods:
        a, b = pd.Timestamp(a), min(pd.Timestamp(b), x1)
        y = rows[ctr]
        colr = {"cap": C[ctr], "tax": "#8C8C8C", "after": C[ctr]}[kind]
        st.barh(y, (b - a).days, left=a, height=0.62, color=colr, alpha=0.35 if kind == "after" else 1.0,
                hatch="///" if kind == "after" else None, edgecolor="white" if kind != "after" else C[ctr], lw=0)
        if label:
            inside = (b - a).days > 150
            near_end = (x1 - b).days < 400                   # short bars at the right edge: label on the left
            if inside:
                pos, ha = a + (b - a) / 2, "center"
            elif near_end:
                pos, ha = a - pd.Timedelta(days=15), "right"
            else:
                pos, ha = b + pd.Timedelta(days=12), "left"
            st.text(pos, y, label, ha=ha, va="center", fontsize=8.5,
                    color="white" if inside and kind != "after" else "#1A1A1A")
    st.set_yticks(range(len(cfg["countries"])), [NAMES[c] for c in cfg["countries"]])
    for lab, c in zip(st.get_yticklabels(), cfg["countries"]):
        lab.set_color(C[c])
        lab.set_fontweight("bold")
    st.set_ylim(len(cfg["countries"]) - 0.4, -0.6)
    st.grid(False)
    st.spines["left"].set_visible(False)
    st.tick_params(axis="y", length=0)
    st.set_title("Measures in force", loc="left", fontsize=10)
    st.legend([Patch(color="#404040"), Patch(color="#8C8C8C"), Patch(facecolor="#BFBFBF", hatch="///")],
              ["price cap / margin cap", "tax cut", "after the cap (margin premium)"], loc="lower right",
              bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=8.5)
    st.set_xlim(x0, x1)
    st.xaxis.set_major_locator(mdates.YearLocator())
    st.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save(fig, "24_intervention_timeline",
         "Caps and fuel tax changes (vs. Jan 2021) combined; direct effect only.")


def fig_intervention_timeline(cfg, tl):
    """All fuel interventions since 2021: direct effect on headline inflation, caps vs tax changes, labelled."""
    tl = tl.assign(date=pd.to_datetime(tl.date))
    labels = cfg["charts"].get("timeline_labels", [])
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.2), sharex=True, sharey=True)
    for ax, ctr in zip(axes, cfg["countries"]):
        d = tl[tl.country == ctr].set_index("date").sort_index()
        caps, taxes = d.caps_pp.fillna(0), d.taxes_pp.fillna(0)
        ax.bar(d.index, caps, width=26, color=C[ctr], label="Price caps, margin caps")
        ax.bar(d.index, taxes, width=26, bottom=np.where((taxes >= 0) == (caps >= 0), caps, 0),
               color="#A6A6A6", label="Tax changes (VAT, excise)")
        ax.axhline(0, color=COLORS["grey"], lw=0.8)
        ax.text(0.005, 0.92, NAMES[ctr], transform=ax.transAxes, fontweight="bold", fontsize=11, color=C[ctr],
                va="top")
        for c_, month, text in labels:
            if c_ != ctr:
                continue
            t = pd.Timestamp(month)
            v = d.total_pp.get(t, 0.0)
            ax.annotate(text, xy=(t, v), xytext=(0, 6 if v >= 0 else -6), textcoords="offset points",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color="#1A1A1A")
        ax.set_ylabel("pp")
        ax.grid(axis="x", visible=False)
    lo, hi = tl.total_pp.min(), tl.total_pp.max()
    axes[0].set_ylim(lo - 0.6, hi + 0.6)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    from matplotlib.patches import Patch
    fig.legend([Patch(color="#404040"), Patch(color="#A6A6A6")],
               ["Price caps, margin caps (country colour)", "Tax changes (VAT, excise)"],
               loc="upper right", bbox_to_anchor=(0.99, 0.965), ncol=2, fontsize=9)
    fig.suptitle("Direct effect of fuel interventions on headline inflation (y/y, pp)", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "24b_intervention_bars",
         "Caps vs. Stage 1 counterfactual; taxes vs. Jan 2021 level; direct effect only.", note_y=-0.01)


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
         "Direct effect only; positive after a measure ends (base effect).")


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
         f"Cross-check: {NAMES[c['country']]}'s margin over crude vs. {' and '.join(NAMES[k] for k in c['controls'])}. "
         "Shaded: cap and post-cap window.")


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
    table_title(fig, t, f"Response to a {cfg['stage2']['scale']}% oil price rise, %: before 2021, 2008-26 "
                "(incl. 2021-23 and 2026), and two robustness checks on 2008-26", gap_pt=24)
    save(fig, "13_sample_comparison",
         "Shaded: 2008-26 outside the 90% band of 2008-21. Gas fixed: EU gas price added. Const. taxes: HICP at constant tax rates.", note_y=note_below(fig, t),
         note_x=table_left(fig, t))


def fig_chain_combined(cfg, ch, pairing, name, horizons=(1, 3, 12)):
    """The pass-through chain: price steps (top) and contributions to headline (bottom), per country."""
    d = ch[ch.pairing == pairing]
    get = lambda ctr, step, source: d[(d.country == ctr) & (d.step == step) & (d.source == source)].set_index("horizon")
    k = cfg["stage2"]["scale"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    x = np.arange(len(horizons))
    for col, ctr in enumerate(cfg["countries"]):
        top, bottom = axes[0, col], axes[1, col]
        bars = [("Brent in local currency", get(ctr, "brent_lcu_pct", "stage2"), "#A6A6A6", True),
                ("Pump price incl. tax, predicted from Stage 1", get(ctr, "hicp_fuels_pct", "predicted"),
                 COLORS["crude"], False),
                ("HICP fuels, estimated (Stage 2)", get(ctr, "hicp_fuels_pct", "stage2"), COLORS["crude"], True)]
        for j, (lab, s, colr, filled) in enumerate(bars):
            v = s.loc[list(horizons)]
            top.bar(x + (j - 1) * 0.27, v.value, width=0.26, color=colr if filled else "white", edgecolor=colr,
                    lw=1.5, label=lab if col == 0 else None)
            if v.lower.notna().all():
                top.errorbar(x + (j - 1) * 0.27, v.value, yerr=[v.value - v.lower, v.upper - v.value],
                             fmt="none", color="black", lw=0.8, capsize=2)
        top.set_xticks(x, [f"{h}m" for h in horizons])
        top.set_title(NAMES[ctr], color=C[ctr])
        top.grid(axis="x", visible=False)

        for i, h in enumerate(horizons):
            pos = neg = 0.0
            for part, lab, colr in CONTRIB_PARTS:
                v = get(ctr, "direct_fuels_pp" if part == "fuels" else f"{part}_pp", "stage2").value[h]
                bottom.bar(i, v, bottom=pos if v >= 0 else neg, width=0.55, color=colr,
                           label=lab if (col, i) == (0, 0) else None)
                pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
            t = get(ctr, "total_pp", "stage2").loc[h]
            bottom.errorbar(i, t.value, yerr=[[t.value - t.lower], [t.upper - t.value]], fmt="D", color="black",
                            ms=5, capsize=3, lw=1, label="Headline total, 90% band" if (col, i) == (0, 0) else None)
            pred = get(ctr, "direct_fuels_pp", "predicted").value[h]
            bottom.plot([i - 0.33, i + 0.33], [pred, pred], color=COLORS["accent"], lw=2,
                        label="Fuels in headline, predicted from Stage 1" if (col, i) == (0, 0) else None)
        bottom.axhline(0, color=COLORS["grey"], lw=0.8)
        bottom.set_xticks(x, [f"{h}m" for h in horizons])
        bottom.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel(f"% change per {k}% oil rise")
    axes[1, 0].set_ylabel("pp of headline HICP")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94),
               fontsize=8)
    fig.legend(*axes[1, 0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               fontsize=8)
    pair = cfg["chain"]["pairs"][pairing]
    period = {"pre2020": "2008-19", "pre2021": "2008-21", "main": "2008-26"}
    fig.suptitle(f"From crude oil to headline inflation: Stage 1 ({period[pair[0]]}) and Stage 2 "
                 f"({period[pair[1]]}) combined", x=0.01, ha="left", fontweight="bold")
    save(fig, name, "Predicted: Stage 1 path x crude share x fuel weight. Estimated: Stage 2; 90% bands.",
         note_y=-0.03, rect=(0, 0.07, 1, 0.88))


def fig_components_3_12(cfg, s2, sample="pre2021", horizons=(3, 12)):
    """Immediate vs delayed: headline response split into components at 3 and 12 months, values labelled."""
    c = s2[(s2.spec == "contribution") & (s2["sample"] == sample)]
    k = cfg["stage2"]["scale"]
    width, gap = 0.7, 0.9
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ticks, labels = [], []
    for i, ctr in enumerate(cfg["countries"]):
        for j, h in enumerate(horizons):
            x = i * (len(horizons) + gap) + j
            d = c[(c.country == ctr) & (c.horizon == h)].set_index("measure").estimate
            pos = neg = 0.0
            for part, lab, colr in CONTRIB_PARTS:
                v = d[f"contrib_{part}"]
                base = pos if v >= 0 else neg
                ax.bar(x, v, bottom=base, color=colr, width=width, label=lab if (i, j) == (0, 0) else None)
                if abs(v) >= 0.04:
                    ax.text(x, base + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                            color="white" if part in ("fuels", "food") else "black")
                pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
            ax.text(x, pos + 0.012, f"{d['contrib_total']:.2f}", ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold")
            ticks.append(x)
            labels.append(f"{h} months")
        ax.text(i * (len(horizons) + gap) + (len(horizons) - 1) / 2, -0.075, NAMES[ctr], ha="center", va="top",
                fontsize=10.5, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_xticks(ticks, labels, fontsize=9)
    ax.axhline(0, color=COLORS["grey"], lw=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("pp of headline HICP")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    period = {"pre2021": "2008-21", "main": "2008-26"}[sample]
    ax.set_title(f"Headline response to a {k}% Brent increase: immediate vs delayed, by component ({period})",
                 loc="left")
    save(fig, "23_components", "Contributions with year-specific HICP weights; parts add up to the total (bold).", note_y=-0.1)


def fig_components_over_time(cfg, s2, sample="pre2021"):
    """Headline response by month, split into contributions: how the composition shifts from fuels to food."""
    c = s2[(s2.spec == "contribution") & (s2["sample"] == sample)]
    k = cfg["stage2"]["scale"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, ctr in zip(axes, cfg["countries"]):
        d = c[c.country == ctr].pivot_table(index="horizon", columns="measure", values="estimate").sort_index()
        pos = np.zeros(len(d))
        neg = np.zeros(len(d))
        for part, lab, colr in CONTRIB_PARTS:
            vals = d[f"contrib_{part}"].values
            ax.bar(d.index, vals, bottom=np.where(vals >= 0, pos, neg), color=colr, width=0.75,
                   label=lab if ctr == cfg["countries"][0] else None)
            pos += np.clip(vals, 0, None)
            neg += np.clip(vals, None, 0)
        ax.plot(d.index, d["contrib_total"], color="black", lw=1.5, marker="o", ms=3.5,
                label="Headline, total" if ctr == cfg["countries"][0] else None)
        ax.axhline(0, color=COLORS["grey"], lw=0.8)
        ax.set_xticks([0, 3, 6, 9, 12])
        ax.set_xlabel("months after the oil price increase")
        ax.set_title(NAMES[ctr])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("pp of headline HICP")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0),
               fontsize=8.5)
    period = {"pre2021": "2008-21", "main": "2008-26"}[sample]
    fig.suptitle(f"Headline response to a {k}% Brent increase by component, {period}", x=0.01, ha="left",
                 fontweight="bold")
    save(fig, "23b_components_by_month", "Contributions with year-specific HICP weights; parts add up to the total.", note_y=-0.03, rect=(0, 0.08, 1, 0.93))


def fig_pump_to_headline(cfg, ch, example="HU", h=3, pairing="normal_times"):
    """From pump prices to headline: the logic (flow for one country) and the result (all countries)."""
    d = ch[(ch.pairing == pairing) & (ch.horizon == h)]
    v = lambda ctr, step, source: d[(d.country == ctr) & (d.step == step) & (d.source == source)].iloc[0]
    k = cfg["stage2"]["scale"]
    cur = cfg["currencies"][example]

    fig = plt.figure(figsize=(11, 6.4))
    top = fig.add_axes([0.02, 0.62, 0.96, 0.30])
    bottom = fig.add_axes([0.08, 0.08, 0.88, 0.42])

    # --- logic: worked example, two routes to the same direct effect ---
    top.set_xlim(0, 1)
    top.set_ylim(3.55, 0)
    top.axis("off")
    brent = v(example, "brent_lcu_pct", "stage2").value
    pump = v(example, "hicp_fuels_pct", "predicted").value
    pred = v(example, "direct_fuels_pp", "predicted").value
    est = v(example, "direct_fuels_pp", "stage2").value
    weight = pred / pump * 100
    dark, grey = "#1A1A1A", COLORS["grey"]
    top.text(0.0, 0.25, f"How the direct effect is calculated: {NAMES[example]}, {h} months after a {k}% Brent increase",
             fontsize=11, fontweight="bold", va="center")
    top.text(0.93, 0.85, "Fuels in headline", ha="center", va="center", fontsize=9, color=grey)
    top.plot([0, 1], [1.1, 1.1], color=grey, lw=1.0)
    # route 1
    y1 = 1.65
    top.text(0.0, y1, "Route 1", fontsize=10.5, fontweight="bold", va="center", color=dark)
    top.text(0.0, y1 + 0.42, "from pump prices (Stage 1)", fontsize=8.5, va="center", color=grey)
    steps = [(0.33, f"Brent in {cur} +{brent:.1f}%", "oil price move in local currency"),
             (0.53, f"pump price +{pump:.1f}%", "passed on ~1:1; crude ~1/3 of price"),
             (0.73, f"x fuel weight {weight:.1f}%", "share of fuels in the basket")]
    for i, (x, txt, note) in enumerate(steps):
        top.text(x, y1, txt, ha="center", va="center", fontsize=10.5, color=dark)
        top.text(x, y1 + 0.42, note, ha="center", va="center", fontsize=8, color=grey)
        if i:
            top.text((x + steps[i - 1][0]) / 2, y1, "\u2192", ha="center", va="center", fontsize=11, color=grey)
    top.text(0.84, y1, "=", ha="center", va="center", fontsize=11, color=grey)
    top.text(0.93, y1, f"{pred:.2f} pp", ha="center", va="center", fontsize=11.5, fontweight="bold", color=dark)
    top.plot([0, 1], [2.35, 2.35], color=COLORS["light"], lw=0.8)
    # route 2
    y2 = 2.8
    top.text(0.0, y2, "Route 2", fontsize=10.5, fontweight="bold", va="center", color=dark)
    top.text(0.0, y2 + 0.42, "from HICP data (Stage 2)", fontsize=8.5, va="center", color=grey)
    steps2 = [(0.33, "fuel contribution", "fuel price change x weight, monthly"),
              (0.53, "regressed on Brent", f"local projection, {h} months ahead"),
              (0.73, "controls held fixed", "exchange rate, demand, taxes, caps")]
    for i, (x, txt, note) in enumerate(steps2):
        top.text(x, y2, txt, ha="center", va="center", fontsize=10.5, color=dark)
        top.text(x, y2 + 0.42, note, ha="center", va="center", fontsize=8, color=grey)
        if i:
            top.text((x + steps2[i - 1][0]) / 2, y2, "\u2192", ha="center", va="center", fontsize=11, color=grey)
    top.text(0.84, y2, "=", ha="center", va="center", fontsize=11, color=grey)
    top.text(0.93, y2, f"{est:.2f} pp", ha="center", va="center", fontsize=11.5, fontweight="bold", color=dark)
    top.plot([0, 1], [3.5, 3.5], color=grey, lw=1.0)

    # --- result: all countries ---
    x = np.arange(len(cfg["countries"]))
    for i, ctr in enumerate(cfg["countries"]):
        tot = v(ctr, "total_pp", "stage2")
        p_ = v(ctr, "direct_fuels_pp", "predicted")
        e_ = v(ctr, "direct_fuels_pp", "stage2")
        bottom.bar(i, tot.value, width=0.62, color="white", edgecolor=COLORS["grey"], lw=1.2, zorder=1,
                   label="Headline, total" if i == 0 else None)
        bottom.bar(i - 0.14, p_.value, width=0.26, color="white", edgecolor=C[ctr], lw=1.8, hatch="///", zorder=2,
                   label="Fuels: predicted from pump prices (Stage 1)" if i == 0 else None)
        bottom.bar(i + 0.14, e_.value, width=0.26, color=C[ctr], zorder=2,
                   label="Fuels: estimated from HICP data (Stage 2)" if i == 0 else None)
        bottom.errorbar(i + 0.14, e_.value, yerr=[[e_.value - e_.lower], [e_.upper - e_.value]], fmt="none",
                        color="black", lw=0.9, capsize=3, zorder=3)
        bottom.text(i, tot.value + 0.012, f"{tot.value:.2f}", ha="center", va="bottom", fontsize=9,
                    color=COLORS["grey"])
        for xx, val in ((i - 0.14, p_.value), (i + 0.14, e_.value)):
            bottom.text(xx, val / 2, f"{val:.2f}", ha="center", va="center", fontsize=8.5,
                        color="black" if xx < i else "white", fontweight="bold", zorder=4)
    bottom.set_xticks(x, [NAMES[c] for c in cfg["countries"]])
    bottom.set_ylabel("pp of headline HICP")
    bottom.grid(axis="x", visible=False)
    leg = bottom.legend(loc="upper left", fontsize=8.5)
    for patch in leg.get_patches():
        patch.set_edgecolor(COLORS["grey"])
        if patch.get_facecolor()[:3] != (1.0, 1.0, 1.0):
            patch.set_facecolor(COLORS["grey"])
    bottom.set_title(f"All countries: direct effect of fuels on headline, {h} months after a {k}% Brent increase",
                     loc="left", fontsize=11)
    fig.suptitle("From pump prices to headline inflation", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.text(0.01, -0.02, f"{SOURCE} Normal times: Stage 1 2008-19, Stage 2 2008-21; 90% band.", fontsize=7.5, color=COLORS["grey"],
             ha="left", va="top", wrap=True)
    fig.savefig(utils.FIGURES / "22_pump_to_headline.png", bbox_inches="tight")
    plt.close(fig)
    print("    22_pump_to_headline.png")


def energy_table(cfg, monthly, s2, horizons=(3, 12)):
    """Contribution of each energy item to the headline response, with its basket weight."""
    items = [("petrol", "from2015", "Petrol (2015-26)"), ("diesel", "from2015", "Diesel (2015-26)"),
             ("household_gas", "main", "Household gas"), ("electricity", "main", "Electricity"),
             ("heating_other", "main", "Heating, solid and liquid fuels"),
             ("household_gas", "pre2021", "Household gas, 2008-21"),
             ("heating_other", "pre2021", "Heating, solid and liquid fuels, 2008-21")]
    c = s2[s2.spec == "contribution"]
    rows, cells = [], []
    for part, sample, label in items:
        row, cell = {"item": label}, []
        for ctr in cfg["countries"]:
            lw = monthly[(monthly.country == ctr) & (monthly.month == monthly.month.max())].set_index("component").weight
            weight = {"heating_other": lw["energy"] - lw["fuels"] - lw["electricity"] - lw["household_gas"]}.get(
                part, lw.get(part, np.nan))
            row[f"{NAMES[ctr]} weight 2026"] = round(weight, 1)
            cell.append(f"{weight:.0f}")
            for h in horizons:
                d = c[(c.country == ctr) & (c["sample"] == sample) & (c.horizon == h)
                      & (c.measure == f"contrib_{part}")].iloc[0]
                row[f"{NAMES[ctr]} {h}m"] = round(d.estimate, 2)
                row[f"{NAMES[ctr]} {h}m band"] = f"[{d.lower:.2f}, {d.upper:.2f}]"
                cell.append(f"{d.estimate:.2f}")
        rows.append(row)
        cells.append(cell)
    pd.DataFrame(rows).to_csv(utils.TABLES / "energy_contributions.csv", index=False)
    print("    energy_contributions.csv")

    n = 1 + len(horizons)
    fig, ax = plt.subplots(figsize=(13, 0.22 * len(cells) + 1.0))
    ax.axis("off")
    col_labels = [lab for _ in cfg["countries"] for lab in ["weight, ‰"] + [f"{h}m" for h in horizons]]
    t = ax.table(cellText=cells, rowLabels=[r["item"] for r in rows], colLabels=col_labels,
                 loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.35)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor(COLORS["light"])
        if i == 0 or j == -1:
            cell.set_text_props(fontweight="bold")
        if j >= 0 and j % n == 0 and i > 0:
            cell.set_text_props(color=COLORS["grey"])
    fig.canvas.draw()
    for k, ctr in enumerate(cfg["countries"]):
        a, b = t[0, n * k].get_window_extent(), t[0, n * k + n - 1].get_window_extent()
        x = ax.transAxes.inverted().transform(((a.x0 + b.x1) / 2, a.y1))
        ax.text(x[0], x[1] + 0.01, NAMES[ctr], transform=ax.transAxes, ha="center", va="bottom",
                fontweight="bold", color=C[ctr])
    table_title(fig, t, f"Energy items: contribution to headline inflation after a {cfg['stage2']['scale']}% oil "
                "price rise, pp", gap_pt=24)
    save(fig, "16_energy_contributions",
         "Contributions with year-specific HICP weights; petrol and diesel from 2015. Weight: 2026, per mille.",
         note_y=note_below(fig, t), note_x=table_left(fig, t))


def round_to_100(shares):
    """Round percentages to whole numbers that still sum to 100 (largest remainder)."""
    floor = np.floor(shares).astype(int)
    order = np.argsort(-(shares - floor))
    floor[order[:100 - floor.sum()]] += 1
    return floor


def price_structure_table(cfg, weekly, start="2008-01-01"):
    """Pump price = 100%: crude, margin, per-litre taxes and VAT, petrol and diesel; average from `start`,
    weeks with a price cap, margin cap or discount left out (they distort the structure)."""
    parts = [("Crude oil", lambda g: g.brent_lcu_prev_week),
             ("Margin (refining, retail)", lambda g: g.price_pre_tax_lcu - g.brent_lcu_prev_week),
             ("Excise (per litre)", lambda g: g.fixed_taxes_lcu),
             ("VAT", lambda g: g.price_with_tax_lcu - g.price_pre_tax_lcu - g.fixed_taxes_lcu)]
    fuels = cfg["stage1"]["fuels"]
    cols, periods = {}, {}
    for ctr in cfg["countries"]:
        for fuel in fuels:
            g = weekly[(weekly.country == ctr) & (weekly.fuel == fuel) & ~weekly.in_intervention].sort_values("date")
            g = g[g.date >= start]
            price = g.price_with_tax_lcu.mean()
            cols[(ctr, fuel)] = round_to_100(np.array([100 * f(g).mean() / price for _, f in parts]))
    labels = [p for p, _ in parts] + ["Pump price"]
    table = pd.DataFrame({f"{NAMES[c]} {f}": list(v) + [100] for (c, f), v in cols.items()}, index=labels)
    table.to_csv(utils.TABLES / "price_structure.csv")
    print("    price_structure.csv")

    cells = [[f"{v}%" for v in row] for row in table.values]
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(cells) + 1.0))
    ax.axis("off")
    t = ax.table(cellText=cells, rowLabels=labels, colLabels=[f.capitalize() for _ in cfg["countries"] for f in fuels],
                 loc="upper center", cellLoc="center", colWidths=[0.12] * len(cols))
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.5)
    n = len(fuels)
    for (i, j), cell in t.get_celld().items():
        cell.set_edgecolor(COLORS["light"])
        if i == 0 or j == -1:
            cell.set_text_props(fontweight="bold")
        if i == len(cells):                                  # pump price row
            cell.set_facecolor("#F2F2F2")
            cell.set_text_props(fontweight="bold")
    fig.canvas.draw()
    for k, ctr in enumerate(cfg["countries"]):
        a, b = t[0, n * k].get_window_extent(), t[0, n * k + n - 1].get_window_extent()
        x = ax.transAxes.inverted().transform(((a.x0 + b.x1) / 2, a.y1))
        ax.text(x[0], x[1] + 0.01, NAMES[ctr], transform=ax.transAxes, ha="center", va="bottom",
                fontweight="bold", color=C[ctr])
    table_title(fig, t, f"Pump price structure, % of retail price, {start[:4]}-{weekly.date.max():%Y} average",
                gap_pt=24)
    save(fig, "18_price_structure",
         "Weeks with price measures excluded. Margin: refining, transport, retail.",
         note_y=note_below(fig, t), note_x=table_left(fig, t))


def crude_share_table(cfg, weekly, fuel="petrol", start="2008-01-01"):
    """Share of crude oil in the pump price incl. taxes, by year (weeks with a price measure left out)."""
    g = weekly[(weekly.fuel == fuel) & (weekly.date >= start) & ~weekly.in_intervention].copy()
    g["year"] = g.date.dt.year
    sums = g.groupby(["country", "year"])[["brent_lcu_prev_week", "price_with_tax_lcu"]].mean()
    share = (100 * sums.brent_lcu_prev_week / sums.price_with_tax_lcu).unstack("year").reindex(cfg["countries"])
    share.index = [NAMES[c] for c in share.index]
    share.round(1).to_csv(utils.TABLES / f"crude_share_{fuel}.csv")
    print(f"    crude_share_{fuel}.csv")

    years = list(share.columns)
    last = weekly.date.max()
    labels = [f"'{y % 100:02d}" + ("*" if y == last.year else "") for y in years]
    cells = [["cap" if np.isnan(v) else f"{v:.0f}" for v in row] for row in share.values]
    fig, ax = plt.subplots(figsize=(10, 1.6))
    ax.axis("off")
    t = ax.table(cellText=cells, rowLabels=list(share.index), colLabels=labels, loc="upper center",
                 cellLoc="center", rowLoc="left", colWidths=[0.047] * len(labels))
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.6)
    for (i, j), cell in t.get_celld().items():               # minimal: horizontal rules only, bold header
        cell.set_edgecolor(COLORS["grey"] if i == 0 else COLORS["light"])
        cell.visible_edges = "B" if i < len(cells) else ""
        if i == 0:
            cell.set_text_props(fontweight="bold")
        if j == -1:
            cell.set_text_props(ha="left")
    table_title(fig, t, f"Share of crude oil in the {fuel} price incl. taxes, %, annual average")
    save(fig, f"19_crude_share_{fuel}",
         f"Weeks with price measures excluded ('cap': all year). *{last.year}: January to {last:%B}.", note_y=note_below(fig, t),
         note_x=table_left(fig, t))


def long_run_table(cfg, s1):
    """Long-run pass-through with 95% band, 2008-19 and 2008-26; country rows split into petrol and diesel."""
    samples = [("pre2020", "2008-19"), ("main", "2008-26")]
    cells, labels, header_rows, rows = [], [], [], []
    for ctr in cfg["countries"]:
        header_rows.append(len(cells))
        cells.append([""] * len(samples))
        labels.append(NAMES[ctr])
        for fuel in cfg["stage1"]["fuels"]:
            row, out = [], {"country": NAMES[ctr], "fuel": fuel}
            for sample, lab in samples:
                d = s1_main(s1, sample=sample)
                r = d[(d.country == ctr) & (d.fuel == fuel) & (d.regime == "normal") & (d.measure == "long_run")].iloc[0]
                row.append(f"{r.estimate:.2f} [{r.lower:.2f}, {r.upper:.2f}]")
                out.update({lab: round(r.estimate, 3), f"{lab} lower": round(r.lower, 3), f"{lab} upper": round(r.upper, 3)})
            cells.append(row)
            labels.append("   " + fuel.capitalize())
            rows.append(out)
    pd.DataFrame(rows).to_csv(utils.TABLES / "long_run.csv", index=False)
    print("    long_run.csv")

    # drawn by hand for typographic control: estimate in dark type, interval in smaller grey type
    n_rows = len(cells)
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * n_rows + 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(n_rows + 0.2, -1.4)
    ax.axis("off")
    col_x = [0.50, 0.80]
    sub = {"2008-19": "normal times", "2008-26": "incl. 2020-26 crises"}
    for x, (_, lab) in zip(col_x, samples):
        ax.text(x, -0.95, lab, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x, -0.45, sub[lab], ha="center", va="center", fontsize=8.5, color=COLORS["grey"])
    ax.plot([0, 1], [-0.1, -0.1], color=COLORS["grey"], lw=1.2)
    for i, (label, row) in enumerate(zip(labels, cells)):
        y = i + 0.45
        if i in header_rows:
            ax.text(0.0, y, label, ha="left", va="center", fontsize=10.5, fontweight="bold")
            if i > 0:
                ax.plot([0, 1], [i - 0.05, i - 0.05], color=COLORS["light"], lw=0.8)
            continue
        ax.text(0.04, y, label.strip(), ha="left", va="center", fontsize=10)
        for x, txt in zip(col_x, row):
            est, ci = txt.split(" ", 1)
            ax.text(x - 0.012, y, est, ha="right", va="center", fontsize=11, color="#1A1A1A")
            ax.text(x - 0.002, y, ci, ha="left", va="center", fontsize=8.5, color=COLORS["grey"])
    ax.plot([0, 1], [n_rows + 0.05, n_rows + 0.05], color=COLORS["grey"], lw=1.2)
    ax.set_title("Long-run pass-through from crude to pre-tax pump prices", loc="left", pad=6, fontsize=11.5)
    save(fig, "21_long_run",
         "Dynamic OLS, weeks without price measures; 95% confidence intervals in brackets.",
         note_y=0.09, note_x=ax.get_position().x0)


def driver_table(cfg, weekly, monthly, s1, s2, recent_weeks=52):
    """Why the countries differ: structural drivers next to the outcomes they explain."""
    m2, m1 = s2_main(s2), s1_main(s1, sample="pre2020")
    contrib = s2[s2.spec == "contribution"]
    start = cfg["stage2"]["samples"]["main"]["start"]
    sections = ["1. Crude in local currency", "2. Pump price before tax", "3. Pump price with tax",
                "4. HICP fuels", "5. Direct effect on headline", "6. Indirect effects", "Headline", "Policy"]
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
        version = s2[s2.main_estimate & (s2.country == ctr)].policy_handling.iloc[0]

        def path(comp, sample="pre2021"):
            d = s2[(s2.country == ctr) & (s2.component == comp) & (s2["sample"] == sample) & (s2.spec == "baseline")
                   & (s2.regime == "normal") & (s2.measure == "response") & (s2.policy_handling == version)]
            return d.sort_values("horizon").set_index("horizon").estimate

        fuels_path, head_path = path("fuels"), path("headline")
        c3 = contrib[(contrib.country == ctr) & (contrib["sample"] == "main") & (contrib.horizon == 3)]
        c3 = c3.set_index("measure").estimate
        values[ctr] = {
            "1. Crude in local currency": {
                "HICP fuels after 1 month, % per 10% weaker currency vs USD": f"{resp('fuels', 1, 'fx_response'):.1f}",
            },
            "2. Pump price before tax": {
                "Strength: pass-through after 4 weeks (2008-19)": f"{s1v('cumulative', 4):.2f}",
                "Speed: weeks to 90% of long run (2008-19)": f"{s1v('weeks_to_90pct'):.1f}",
            },
            "3. Pump price with tax": {
                "Crude share of pump price incl. VAT, % (last 52 wks)":
                    f"{100 * ((1 + w.vat_pct / 100) * w.brent_lcu_prev_week / w.price_with_tax_lcu).mean():.0f}",
                "Per-litre taxes (excise etc.), % of pump price": f"{100 * (w.fixed_taxes_lcu / w.price_with_tax_lcu).mean():.0f}",
                "VAT on fuel, %": f"{w.vat_pct.iloc[-1]:.0f}",
            },
            "4. HICP fuels": {
                "Strength: HICP fuels after 2 months, % per 10% oil": f"{resp('fuels', 2):.1f}",
                "Speed: month of peak response (2008-21)": f"{int(fuels_path.idxmax())}",
            },
            "5. Direct effect on headline": {
                "Fuel weight in HICP, per mille (2026 / avg 2008-26)":
                    f"{latest['fuels']:.0f} / {hist[hist.component == 'fuels'].drop_duplicates('month').weight.mean():.0f}",
                "Petrol share of fuel weight, % (2026)": f"{100 * latest['petrol'] / (latest['petrol'] + latest['diesel']):.0f}",
                "Fuels in headline after 3 months, pp": f"{c3['contrib_fuels']:.2f}",
            },
            "6. Indirect effects": {
                "Food weight / administered-price weight, per mille (2026)":
                    f"{latest['food']:.0f} / {latest['administered']:.0f}",
                "Food / core in headline after 12 months, pp (2008-26)":
                    f"{c12['contrib_food']:.2f} / {c12['contrib_core']:.2f}",
            },
            "Headline": {
                "Strength: headline after 3 months, % per 10% oil": f"{resp('headline', 3):.2f}",
                "Strength: headline after 12 months, 2008-21 / 2008-26": f"{pre:.2f} / {resp('headline', 12):.2f}",
                "Speed: % of peak response reached after 3 months (2008-21)":
                    f"{100 * head_path[3] / head_path.max():.0f}",
            },
            "Policy": {
                "Months with a cap, margin cap or discount, 2008-26":
                    f"{int((hist.drop_duplicates('month').intervention_share > 0).sum())}",
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
    table_title(fig, t, "Why pass-through differs across Poland, Romania and Hungary, step by step along the chain")
    save(fig, "15_country_drivers",
         "Price structure: last 52 weeks without price measures. Currency benchmark: about 4%.", note_y=note_below(fig, t),
         note_x=table_left(fig, t))


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
    fig_crude_to_pump(cfg, s1, s1d, markers=True, name="03b_crude_to_pump_differences")
    fig_long_run(cfg, s1)
    long_run_table(cfg, s1)
    fig_policy_stage1(cfg, s1)
    fig_policy_stage1(cfg, s1, ["HU"], "04b_policy_stage1_hungary")
    fig_headline(cfg, s2)
    fig_components(cfg, s2)
    fig_direct_indirect(cfg, s2)
    fig_chain(cfg, s2)
    sample_table(cfg, s2)
    fig_petrol_diesel(cfg, s1, s2)
    driver_table(cfg, weekly, monthly, s1, s2)
    energy_table(cfg, monthly, s2)
    fig_components_3_12(cfg, s2)
    fig_components_over_time(cfg, s2)
    price_structure_table(cfg, weekly)
    crude_share_table(cfg, weekly)
    chain_path = utils.RESULTS / "chain.csv"
    if chain_path.exists():
        ch = pd.read_csv(chain_path)
        fig_chain_combined(cfg, ch, "normal_times", "17_chain_normal_times")
        fig_chain_combined(cfg, ch, "full_period", "17b_chain_full_period")
        fig_pump_to_headline(cfg, ch)
    else:
        print("    note: outputs/results/chain.csv not found - chart 17 skipped (run: python run.py --from chain)")
    pol_path = utils.RESULTS / "policy_counterfactual.csv"
    if pol_path.exists():
        pol = pd.read_csv(pol_path)
        fig_counterfactual_prices(cfg, pol)
        fig_counterfactual_inflation(cfg, pol)
        tl_path = utils.RESULTS / "intervention_timeline.csv"
        if tl_path.exists():
            fig_intervention_lines(cfg, pd.read_csv(tl_path))
            fig_intervention_timeline(cfg, pd.read_csv(tl_path))
        policy_table(pol)
        cc_path = utils.RESULTS / "policy_crosscheck.csv"
        if cc_path.exists():
            fig_crosscheck(cfg, pd.read_csv(cc_path), pol, iv)
    else:
        print("    note: outputs/results/policy_counterfactual.csv not found - charts 09-11 skipped "
              "(run: python run.py --from policy)")
    print(country_table(cfg, weekly, monthly, s1, s2, s1d).to_string())