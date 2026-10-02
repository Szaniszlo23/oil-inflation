"""
fetch.py - download all raw data into data/raw/<YYYY-MM-DD>/.

Sources (codes and URLs come from config.yaml):
  - Brent crude, daily .............. FRED API
  - Exchange rates, daily ........... ECB Data Portal API
  - HICP indices and item weights ... Eurostat Statistics API (JSON-stat)
  - Euro-area industrial production . Eurostat Statistics API (JSON-stat)
  - Pump prices with/without taxes .. EC Weekly Oil Bulletin price history file

Files are saved exactly as received - nothing is cleaned here, that's build.py.
Downloads go into a temporary folder that only gets its final date name once
every source succeeded, so data/raw/<date>/ is always a complete snapshot.
download_log.json records the URL, time, size and checksum of every file.
"""
import hashlib
import json
import re
import shutil
import time
from datetime import datetime, timezone
from urllib.parse import urljoin

import requests

from pipeline import utils

HEADERS = {"User-Agent": "oil-inflation research pipeline"}
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
ECB_API = "https://data-api.ecb.europa.eu/service/data"
EUROSTAT_API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


# --- Helpers -------------------------------------------------------------------

def _hide_key(text):
    """Never write the FRED key into logs or error messages."""
    return re.sub(r"api_key=[^&\s]+", "api_key=***", str(text))


def _get(url, params=None, retries=3):
    """GET with retries on network errors and server errors (5xx)."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=180)
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries:
                raise
            time.sleep(5 * attempt)
            continue
        if r.status_code >= 500 and attempt < retries:
            time.sleep(5 * attempt)
            continue
        if not r.ok:
            raise RuntimeError(f"HTTP {r.status_code} from {_hide_key(r.url)}\n{r.text[:500]}")
        return r


class Snapshot:
    """Collects the downloads of one run and logs where each file came from."""

    def __init__(self, date):
        self.final = utils.DATA_RAW / date
        self.tmp = utils.DATA_RAW / f"{date}_incomplete"
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        self.tmp.mkdir(parents=True)
        self.log = []

    def save(self, name, response, source):
        content = response.content
        (self.tmp / name).write_bytes(content)
        self.log.append({
            "file": name,
            "source": source,
            "url": _hide_key(response.url),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
        print(f"    saved {name} ({len(content) / 1e6:.2f} MB)")

    def finish(self):
        (self.tmp / "download_log.json").write_text(json.dumps(self.log, indent=2))
        if self.final.exists():
            shutil.rmtree(self.final)
        self.tmp.rename(self.final)
        return self.final


# --- Sources -------------------------------------------------------------------

def fetch_fred(snap, cfg):
    s = cfg["series"]["brent"]
    params = {
        "series_id": s["code"],
        "api_key": utils.get_secret("FRED_API_KEY"),
        "file_type": "json",
        "observation_start": s["start"],
    }
    snap.save(f"fred_{s['code']}.json", _get(FRED_API, params), "FRED")


def fetch_ecb(snap, cfg):
    s = cfg["series"]["fx"]
    url = f"{ECB_API}/{s['dataset']}/{s['key']}"
    params = {"format": "csvdata", "startPeriod": s["start"]}
    snap.save("ecb_fx.csv", _get(url, params), "ECB Data Portal")


def _eurostat(dataset, params):
    r = _get(f"{EUROSTAT_API}/{dataset}", {**params, "lang": "en"})
    body = r.json()
    if "error" in body:
        raise RuntimeError(f"Eurostat {dataset}: {body['error']}")
    return r, body


def _check_code(body, dimension, code, dataset):
    """Stop with a readable list of valid codes if config.yaml names one that doesn't exist."""
    if dimension not in body["id"]:
        raise ValueError(f"{dataset} has no dimension '{dimension}'. Dimensions: {body['id']}")
    category = body["dimension"][dimension]["category"]
    if code not in category["index"]:
        labels = category.get("label", {})
        options = "\n".join(f"      {c}: {labels.get(c, '')}" for c in category["index"])
        raise ValueError(f"{dataset}: '{code}' is not a valid {dimension}. Options:\n{options}")


def fetch_eurostat(snap, cfg):
    countries = cfg["countries"]
    idx = cfg["series"]["hicp_index"]

    # 1. Code lists: one recent month, all dimensions. Used to check the configured
    #    unit now and to pin the component codes in config.yaml afterwards.
    r, body = _eurostat(idx["dataset"], {"geo": countries, "lastTimePeriod": 1})
    snap.save(f"eurostat_{idx['dataset']}_codelist.json", r, "Eurostat")
    _check_code(body, "unit", idx["unit"], idx["dataset"])

    # 2. All HICP components, one country per request to keep responses small.
    for geo in countries:
        params = {"geo": geo, "unit": idx["unit"], "sinceTimePeriod": idx["start"]}
        r, _ = _eurostat(idx["dataset"], params)
        snap.save(f"eurostat_{idx['dataset']}_{geo}.json", r, "Eurostat")

    # 3. Item weights for all components.
    w = cfg["series"]["hicp_weights"]
    r, _ = _eurostat(w["dataset"], {"geo": countries, "sinceTimePeriod": w["start"]})
    snap.save(f"eurostat_{w['dataset']}.json", r, "Eurostat")


def fetch_eurostat_ip(snap, cfg):
    """Euro-area industrial production: global-demand control for Stage 2."""
    s = cfg["series"]["ea_industrial_production"]
    r, body = _eurostat(s["dataset"], {"geo": s["geo"], "lastTimePeriod": 1})
    snap.save(f"eurostat_{s['dataset']}_codelist.json", r, "Eurostat")
    for dimension, code in s["filters"].items():
        _check_code(body, dimension, code, s["dataset"])
    r, _ = _eurostat(s["dataset"], {"geo": s["geo"], **s["filters"], "sinceTimePeriod": s["start"]})
    snap.save(f"eurostat_{s['dataset']}.json", r, "Eurostat")


def fetch_oil_bulletin(snap, cfg):
    s = cfg["series"]["oil_bulletin"]
    url = s.get("history_url")
    if not url:
        # No API: find the price-history file on the bulletin page. Its link changes
        # with every update, so we look for it rather than hard-coding it.
        page = _get(s["page"])
        links = [l.replace("&amp;", "&") for l in re.findall(r'href="([^"]+)"', page.text)]
        files = [l for l in links if "download" in l.lower() or ".xlsx" in l.lower()]
        history = [l for l in files if "history" in l.lower()]
        if not history:
            listing = "\n".join(f"      {l}" for l in files) or "      (none)"
            raise RuntimeError(f"No price-history link on {s['page']}. File links found:\n{listing}")
        url = urljoin(s["page"], history[0])
        print(f"    history file found: {url}")
        for other in history[1:]:
            print(f"    (other candidate: {other})")

    r = _get(url)
    if not r.content.startswith(b"PK"):  # .xlsx files are zip archives
        raise RuntimeError(f"Download from {url} is not an Excel file")
    snap.save("oil_bulletin_history.xlsx", r, "EC Weekly Oil Bulletin")


# --- Entry point (called by run.py) ---------------------------------------------

SOURCES = [
    ("Brent crude (FRED)", fetch_fred),
    ("Exchange rates (ECB)", fetch_ecb),
    ("HICP indices and weights (Eurostat)", fetch_eurostat),
    ("Euro-area industrial production (Eurostat)", fetch_eurostat_ip),
    ("Pump prices (EC Weekly Oil Bulletin)", fetch_oil_bulletin),
]


def run(cfg):
    utils.ensure_dirs()
    snap = Snapshot(datetime.now().strftime("%Y-%m-%d"))
    failed = []
    for name, fetch in SOURCES:
        print(f"  - {name}")
        try:
            fetch(snap, cfg)
        except Exception as e:
            print(f"    FAILED: {_hide_key(e)}")
            failed.append(name)

    if failed:
        raise RuntimeError(
            f"Download incomplete ({', '.join(failed)}). "
            f"Partial files kept in {snap.tmp.relative_to(utils.ROOT)} for inspection; "
            f"the last complete snapshot is unchanged."
        )
    path = snap.finish()
    print(f"  Snapshot complete: {path.relative_to(utils.ROOT)}")
    return path