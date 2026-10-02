# Decisions log

Every data and modelling choice, when it was made and why. Status is **decided** or **open**.

---

## 2026-10-02 - First data check (snapshot `data/raw/2026-10-02`)

### D1. HICP source and unit - decided
- **Choice:** Eurostat `prc_hicp_minr`, unit `I25` (index 2025=100), item weights from `prc_hicp_iw`.
- **Why:** From the January 2026 release the HICP uses ECOICOP version 2 and the 2025=100 reference period; the old-classification datasets are archived. The new series are back-cast, so no splicing is needed: headline from 2000-01 for all three countries; components from 2000-01 (PL) and 2000-12 (HU, RO).

### D2. Energy, electricity/gas and core have a break in 2017 - decided (2026-10-02)
- **Finding:** `NRG`, `ELC_GAS` and `TOT_X_NRG_FOOD` are flagged "definition differs" for every month before 2017 and "break in time series" in 2017-01, in all three countries. `TOTAL`, `CP0722` (fuels) and `CP073` (passenger transport services) carry no flags.
- **Evidence:** the January 2017 month-on-month changes are within the normal range of other Januaries in PL and HU (no visible level jump). Romania's energy fell 3.4% in January 2017, but that month also saw a VAT cut on fuel from 20% to 19% (Oil Bulletin VAT sheet), so the drop is policy, not the definition change.
- **Choice:** main estimates over the full sample with a dummy for 2017-01; robustness check from 2017 onwards (about 116 months, still covering 2021-23 and 2026). Headline, fuels and transport services are unaffected. Settings: `stage2.break_month`, `stage2.break_components` in `config.yaml`.

### D3. Petrol and diesel HICP indices only from 2014-12 - decided
- **Choice:** the full-sample fuel analysis uses `CP0722`. `CP07221` (diesel) and `CP07222` (petrol) are used only for the link between Stage 1 pump prices and the HICP, from 2014-12.

### D4. Oil Bulletin prices are in EUR for all countries - decided
- **Finding:** Both price sheets report EUR per 1000 litres, including HU, PL and RO. Backing out the exchange rate the bulletin used (taxes in national currency from the tax sheets vs. the tax gap in EUR, 2024-2026) gives exactly the ECB reference rate of the same Monday: median difference 0.00% for all three countries.
- **Choice:** convert to national currency per litre with the same-day ECB reference rate. The conversion is therefore exact, not an approximation.

### D5. Romania's pump prices start in 2008 - decided
- **Finding:** HU and PL from 2005-01-03, RO from 2008-01-07.
- **Choice:** the Stage 1 cross-country comparison uses a common sample from 2008-01-07 (`stage1.common_start`); the full 2005 sample for HU and PL is a robustness check. Stage 2 does not use bulletin prices, so it runs from 2000-2001 for all three countries.

### D6. Gaps in the weekly Oil Bulletin series - decided (2026-10-02)
- **Finding:** 1,048 weekly steps, plus 25 two-week and 12 three-week gaps, mostly around holidays (37 gap weeks for HU and PL, 30 for RO since 2008).
- **Build:** `build.py` keeps the actual bulletin dates, without interpolation, and records `days_since_previous` for every row.
- **Choice:** `build.py` puts every country and fuel on a regular Monday grid; missing holiday weeks get linearly interpolated pump prices and an `interpolated` flag (49 weeks for HU and PL, 39 for RO). Brent for those Mondays comes from the daily data. Robustness: Stage 1 without any observation whose lag window touches an interpolated week. Interpolated weeks are left out of the tax-sheet check.

### D7. Tax changes come from the Oil Bulletin's own tax sheets - decided (2026-10-02)
- **Finding:** The history file contains dated VAT rates, excise duties and other indirect taxes per country and fuel. Examples: PL cut VAT on fuel to 8% from 2022-02-01 (back to 23% on 2023-01-01) and again from 2026-03-01 (back to 23% on 2026-08-17); PL cut excise from 2026-03-30 to 2026-06-16; RO raised VAT to 21% from 2025-08-01 and cut diesel excise from 2026-07-27 with further changes in September 2026; HU excise changed almost weekly in Aug-Dec 2022 and Mar-Apr 2026, consistent with a rule-based (possibly Brent-linked) mechanism, which still needs confirming in Hungarian law.
- **Choice:** tax rates and their dates come from these sheets (data, downloaded automatically); `interventions.csv` holds only measures that are not in them: price caps, subsidies, regulated prices. Ground rule 3 in `structure.md` updated accordingly.

### D8. Brent 2026 path - decided
- **Finding:** FRED `DCOILBRENTEU` peaked at 138.21 USD/bbl on 2026-04-07 (matches the ECB blog figure; an earlier estimate of about 118 was wrong). Monthly averages: Mar 103, Apr 117, May 107, Jun 85, Jul 84, Aug 91, Sep 114 (peak 130.8). September 2026 is a second spike.

### D9. Oil Bulletin history link - decided
- **Choice:** not pinned in `config.yaml`; `fetch.py` finds it on the bulletin page each time because the link changes with every update. The URL used is recorded in each snapshot's `download_log.json`.

### D10. Exchange rates - decided
- **Finding:** ECB daily rates from 2000-01-03 to 2026-10-01; HUF, PLN and USD have 61 empty rows (days without a fixing). These are dropped in `build.py`.

---

## 2026-10-02 - Build stage

### D11. HICP weights are year-specific; fuel response and fuel weight are reported separately - decided
- **Finding:** transport-fuel weights (per mille), 2026: HU 58.3, RO 48.0, PL 32.7. Hungary's weight moved a lot over time (48.8 in 2005, 81.7 in 2015). Hungary's basket is mostly petrol (51.3 petrol vs 6.6 diesel), PL 19.3/10.5, RO 28.7/14.8.
- **Choice:** contributions use the weight of the year in question. Results separate how strongly fuel prices respond (Stage 1) from how much fuel weighs in the basket (headline effect = weight x fuel response + indirect effects).
- **Hypothesis to check, not to state:** the HICP covers spending on a country's territory, including by non-residents, so fuel tourism or transit traffic could raise Hungary's weight.

### D12. Brent timing for weekly prices - decided (2026-10-02)
- **Build:** `fuel_weekly` stores Brent both on the bulletin Monday (`brent_*_same_day`) and as the average of the seven days before it (`brent_*_prev_week`), in USD per barrel and local currency per litre.
- **Choice:** the previous-week average. Weekly changes in pre-tax prices on Brent changes (lags 0-3, 2008+, outside interventions) fit clearly better with it in all six country-fuel pairs: R-squared 0.34-0.57 vs 0.28-0.39 with same-day Brent.

### D13. Hungarian price cap and post-cap window - decided
- **Choice:** `retail_price_cap` 2021-11-15 to 2022-12-06 and `post_cap_premium` 2022-12-07 to 2023-09-30, both for petrol and diesel. Berezvai & Helfrich measured the post-cap premium for petrol only; it is applied to diesel too because the mechanism (fewer independent stations, weaker competition) affects both. Robustness: petrol-only post-cap window.
- **Data check:** the bulletin shows Hungarian petrol at about 490 HUF/l in June 2022 against the legal cap of 480 HUF/l, so the cap bound for most but possibly not all sales; to keep in mind when reading the cap period.

### D14. Romania's 2022 pump discount - decided
- **Finding:** 0.50 RON/l off pump prices (0.25 from the state budget, 0.25 voluntary retailer discount), 2022-07-01 to 2022-09-30, extended 2022-10-01 to 2022-12-31. Sources in `interventions.csv`.

### D15. Non-tax measures in 2026 - decided (2026-10-02)
- **Finding:** all three countries intervened in spring 2026. Added to `interventions.csv` with sources:
  - HU: retail price cap ("protected price") 595 HUF/l petrol, 615 HUF/l diesel, 2026-03-10 to 2026-06-26 (law ending it in force 2026-06-27). Applied to Hungarian-registered vehicles; the Oil Bulletin reports prices at about the cap level in this period, so the cap is in the data.
  - PL: daily maximum retail prices (CPN package) 2026-03-31 to 2026-06-30, and again 2026-08-17 to 2026-08-31.
  - RO: margin cap at each operator's 2025 average (OUG 19/2026), crisis period 2026-04-01 to 2026-06-30; renewed by Law 162/2026 (promulgated 2026-08-04) until 2026-10-31. The exact start of the law's application is to confirm (entered as 2026-08-05; one day either way does not matter at weekly frequency).
- **Implication:** spring 2026 is a policy period in all three countries, not a clean test of market pass-through. In September 2026 Brent rose again with the HU and PL caps no longer in force, while RO's margin cap still applies.
- Renamed existing measures with their year (`retail_price_cap_2021`, `post_cap_premium_2023`, `pump_price_discount_2022`) so every episode can get its own dummy.

### D16. Hungary's tax changes are near-weekly in 2022 and 2026 - noted
- **Finding:** `tax_change` is true in 63 Hungarian bulletin weeks since mid-2021, because excise was adjusted almost weekly in Aug-Dec 2022 and Mar-Apr 2026. Stage 1 runs on pre-tax prices, so this mainly matters for retail-price elasticities and for the tax-change dummies.

### D17. The bulletin's tax sheets are misdated in places - decided (2026-10-02)
- **Finding:** the price sheets are internally consistent (taxes = price with tax - price without tax, exact), but the dates in the VAT and excise sheets contradict the prices in several places:
  - PL VAT: the sheet has 8% from 2026-03-01 to 2026-08-16; the prices imply 23% until the end of March, 8% from about 2026-03-31 to 2026-06-30, 23% in July-mid August, 8% for 2026-08-17 to 2026-08-31, then 23%. The prices match the announced CPN dates.
  - RO diesel excise: prices imply a cut to 2.504 RON/l from about early May to end June and a variable rate from late July; the sheet only shows changes from 2026-07-27.
  - HU excise: the sheet shows 142.55 HUF/1000 l x 1000 (petrol) from 2026-07-20; prices imply 139.55 throughout.
  - Smaller gaps before 2026: PL 2024 and RO 2008 imply a VAT about 1-2 points too high, i.e. a tax component missing from the "other taxes" sheet.
- **Choice (automatic, no manual tax rows needed):** `build.py` derives the tax rates embedded in the prices and uses them wherever the sheets disagree:
  - VAT: the sheet rate where consistent with the prices; where the prices clearly imply a different legal rate (gap above 1 point, implied rate within 0.2 points of a rate the country has used), that rate; in unclear transition weeks the previous week's rate is carried forward.
  - Fixed taxes (excise + other per-litre taxes): always from the prices, price with tax / (1 + VAT) - price without tax. The sheet values are kept as `*_sheet` columns for reference.
  - Tax change: VAT changed, or fixed taxes moved by more than 1% in a week (95% of weekly noise is below 0.1%).
  - Every run prints the periods where the sheets disagree with the prices.
- **Result:** Poland's 2026 VAT now reads 23% until end of March, 8% from the first April bulletin to end of June, 23% in July to mid-August, 8% for 17-31 August, 23% from September, matching the announced CPN dates. `interventions.csv` keeps its role (non-tax measures only); the tolerances are in `config.yaml` under `build`.

---

## 2026-10-02 - Stage 1

### D18. Policy periods are modelled, not dropped - decided (David's call)
- **Choice:** the main estimate keeps every week. Normal-times dynamics come from clean weeks (no measure in force, not within 8 weeks after one); the effect of each policy type on the price level, the speed of adjustment and the immediate pass-through is estimated on the weeks with a measure in force, measured against the normal-times model. Policy types are pooled within each country (too few weeks per episode): hard cap / maximum price, margin cap, pump discount, and a level-only post-cap window. `ignore`, `dummies` and `exclude` are comparison versions.
- **Why not one equation with interactions:** the first version allowed only speed and impact to change under a measure; the lagged terms were then partly estimated from capped weeks and pulled the normal-times response down (Hungary: 0.74 vs 0.99 after 4 weeks against the clean benchmark). Measuring policy effects against a market model estimated on market weeks removes that contamination while still using all weeks.
- **Speed measure:** weeks until 50% and 90% of the long-run effect, from the full response path, instead of a half-life of the error-correction term alone (which overstated how slow adjustment is).

### D19. First Stage 1 results - noted
- **Normal times:** about 0.9-1.05 of a crude move reaches the pre-tax pump price within 4 weeks in all six cases; half of it within 1-2 weeks. Consistent with the ECB's "fast and complete in levels".
- **Long run above 1 since 2020, especially diesel:** 2008-2019 long-run pass-through is 0.87-1.04; on the full 2008-2026 sample 1.04-1.35, highest for diesel. Interpretation: in the 2022 and 2026 supply crises refining margins rose together with crude, so crude alone overstates the cost link. To present as a finding, with the pre-2020 numbers as the market benchmark.
- **Ignoring policy distorts most in Hungary:** 4-week pass-through 0.50 (petrol) and 0.65 (diesel) when caps are ignored vs 0.99 and 0.94 in normal times.
- **Cointegration evidence is weak** (Engle-Granger p < 0.05 only for PL diesel; error-correction t-statistics -1.8 to -3.8), consistent with margins shifting in crises. The short-run cumulative pass-through, which is the headline number, does not depend on it.
- **Caps:** Hungary's hard caps dampen pre-tax pass-through (0.66 petrol, 0.86 diesel after 4 weeks) but not to zero: in 2021-22 the retail price stayed near the cap while the excise fell as Brent rose, so part of the shock was absorbed by the state through taxes and part by margins. Poland's maximum prices are formula-based (wholesale price plus a fixed margin), so pass-through continued under them; the Polish estimates are imprecise (about 16 weeks). Romania's margin cap and 2022 discount show no clear dampening.

---

## 2026-10-02 - Stage 2

### D20. Stage 2 design - decided
- **Model:** local projections per country, component and horizon (0-12 months); cumulative % change of the HICP index on the monthly % change in Brent (USD), responses scaled to a 10% rise; exchange rate as a separate regressor; 3 lags; month dummies; 2017 break dummy where relevant (D2); euro-area industrial production as global-demand control for headline and core; Newey-West standard errors with h+1 lags; 90% bands.
- **Main sample from 2008-01** (not 2001 as first planned): the tax control needs the weekly bulletin, which starts in 2008 for Romania, and it keeps Stage 1 and Stage 2 on the same window.
- **Tax control:** the % change in the fuel price caused by tax changes alone (pre-tax price held fixed). The first version controlled for the ratio of the prices with and without tax; that ratio falls mechanically when oil raises the pre-tax price (fixed excise becomes a smaller share), so it absorbed part of the oil effect itself (fuels response 1.5-2.4% instead of about 3.2-4.0%). Caught by the cross-check against Stage 1.
- **Policy:** share of the window under a non-tax measure and its interaction with oil (D18 logic). Estimated only with at least 12 policy months: Hungary has 27; Poland (5) and Romania (10) get a level control only, and no policy effect is reported for them.
- **Windows stay inside each sample:** for samples with an end date, the whole window t..t+h must lie inside it (otherwise later data leak into, e.g., the pre-2021 benchmark).
- **Check on Stage 1:** the pass-through chain Brent in local currency -> pump price with tax -> HICP fuels, estimated with the same specification. The HICP fuels index tracks the bulletin's pump prices almost one-for-one (monthly regression: 0.87-0.97, R-squared 0.91-0.95).

### D21. First Stage 2 results - preliminary (run without the demand control; to update after the full run)
- **Headline, per 10% oil rise:** PL +0.30% after 3 months, +0.83% after 12; RO +0.19% / +0.41%; HU +0.42% / +0.46%. The same order of magnitude as Choi et al. (about 0.4 pp per 10%).
- **Fuels:** +3.2% to +4.0% after 2 months; the chain shows about 85-100% of the full pass-through benchmark (crude share of the pump price x Brent's move in local currency) after 2-3 months, consistent with Stage 1.
- **Direct vs indirect at 12 months:** PL energy 0.30 pp (fuels 0.12), indirect 0.53; RO 0.19 (0.14), indirect 0.23; HU 0.18 (0.17), indirect 0.28. Core reacts in Poland (+0.39% after 12 months, band excludes zero) but not in Romania or Hungary.
- **Hungary's caps:** cut the fuel response after 2 months by 2.8 points (of about 4.0), with no lasting difference after 12 months (catch-up). The headline policy effect at 12 months is positive (+1.2 pp) but confounded with the 2022 inflation surge (gas, forint, food); not to be presented as a policy effect.