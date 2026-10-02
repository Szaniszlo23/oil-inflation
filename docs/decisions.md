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

### D6. Gaps in the weekly Oil Bulletin series - open (build part decided)
- **Finding:** 1,048 weekly steps, plus 25 two-week and 12 three-week gaps, mostly around holidays (37 gap weeks for HU and PL, 30 for RO since 2008).
- **Build:** `build.py` keeps the actual bulletin dates, without interpolation, and records `days_since_previous` for every row.
- **Open for Stage 1:** interpolate single missing weeks onto a regular grid, or estimate on actual dates only.

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

### D12. Brent timing for weekly prices - open
- **Build:** `fuel_weekly` stores Brent both on the bulletin Monday (`brent_*_same_day`) and as the average of the seven days before it (`brent_*_prev_week`), in USD per barrel and local currency per litre.
- **Open for Stage 1:** which one enters the model (the previous-week average avoids reacting to the same day's price moves).

### D13. Hungarian price cap and post-cap window - decided
- **Choice:** `retail_price_cap` 2021-11-15 to 2022-12-06 and `post_cap_premium` 2022-12-07 to 2023-09-30, both for petrol and diesel. Berezvai & Helfrich measured the post-cap premium for petrol only; it is applied to diesel too because the mechanism (fewer independent stations, weaker competition) affects both. Robustness: petrol-only post-cap window.
- **Data check:** the bulletin shows Hungarian petrol at about 490 HUF/l in June 2022 against the legal cap of 480 HUF/l, so the cap bound for most but possibly not all sales; to keep in mind when reading the cap period.

### D14. Romania's 2022 pump discount - decided
- **Finding:** 0.50 RON/l off pump prices (0.25 from the state budget, 0.25 voluntary retailer discount), 2022-07-01 to 2022-09-30, extended 2022-10-01 to 2022-12-31. Sources in `interventions.csv`.

### D15. Non-tax measures in 2026 - open
- **To research before Stage 1:** whether HU, PL or RO introduced price caps, subsidies or regulated fuel prices in 2026 (tax changes are already covered by the bulletin's tax sheets). Only verified measures go into `interventions.csv`.

### D16. Hungary's tax changes are near-weekly in 2022 and 2026 - noted
- **Finding:** `tax_change` is true in 63 Hungarian bulletin weeks since mid-2021, because excise was adjusted almost weekly in Aug-Dec 2022 and Mar-Apr 2026. Stage 1 runs on pre-tax prices, so this mainly matters for retail-price elasticities and for the tax-change dummies.