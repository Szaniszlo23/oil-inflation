# Oil price pass-through to inflation in Poland, Romania and Hungary

How quickly and how strongly changes in crude oil prices reach consumer prices in Poland, Romania and Hungary, and how fuel taxes and price caps change that.

The analysis has two stages:

- **Stage 1 – crude oil to pump prices.** Weekly error-correction model of pre-tax petrol and diesel prices on Brent in local currency (EC Weekly Oil Bulletin, 2008–2026). Gives the long-run pass-through and the speed of adjustment.
- **Stage 2 – crude oil to consumer prices.** Monthly local projections of the HICP headline and its components on Brent (Eurostat, 2008–2026), 0–12 months ahead, controlling for the exchange rate, global demand, tax changes and price caps.

The two stages are linked in a pass-through chain, and price caps and tax changes since 2021 are evaluated against a model counterfactual.

The final presentation is `presentation.pptx`.

## Data

All data is downloaded through public APIs:

| Source | Series |
|---|---|
| FRED | Brent crude (daily), EU natural gas price |
| ECB | Exchange rates (daily) |
| Eurostat | HICP indices and weights, HICP at constant tax rates, euro-area industrial production |
| EC Weekly Oil Bulletin | Petrol and diesel prices with and without taxes, VAT and excise |

Price caps, margin caps and other non-tax measures are listed with their sources in `interventions.csv`.

## Running it

Requires Python 3.10 and a free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "FRED_API_KEY=your_key" > .env

python run.py --refresh      # download a new data snapshot and run everything
python run.py                # rerun on the latest snapshot
python run.py --from stage2  # rerun from a given stage
```

Stages run in this order: `fetch` → `build` → `stage1` → `stage2` → `chain` → `policy` → `charts`.

## Structure

```
config.yaml           all settings: series codes, samples, lags, chart options
interventions.csv     price caps and other non-tax measures, with sources
run.py                entry point
pipeline/
  fetch.py            downloads raw data into data/raw/<date>/
  build.py            cleans and merges into weekly and monthly tables
  stage1.py           crude oil to pump prices (error-correction model)
  stage2.py           crude oil to HICP (local projections)
  chain.py            links Stage 1 and Stage 2
  policy.py           counterfactuals for price caps and tax changes
  charts.py           figures and tables
```

Outputs are written to `outputs/` (results, tables and figures) and are not tracked in git; they can be reproduced with `python run.py`.
