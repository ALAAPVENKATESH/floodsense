"""Download CHIRPS monthly GeoTIFFs and WorldPop population mosaics.

Resumes partial files, skips a file that already has the expected size, and
retries. A blocked or login-gated source prints the manual placement path and
the run continues with whatever is already in data/external/.

Large rasters stay in data/external/ (gitignored). This script writes
data/external/MANIFEST.json, which is the small file meant to be kept.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

EXTERNAL = ROOT / "data" / "external"
CHIRPS_DIR = EXTERNAL / "chirps"
WORLDPOP_DIR = EXTERNAL / "worldpop"
MANIFEST_PATH = EXTERNAL / "MANIFEST.json"
SOURCE_CSV = ROOT / "flood_dataset_classification.csv"

CHIRPS_PAGE = "https://www.chc.ucsb.edu/data/chirps"
CHIRPS_PATTERN = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/"
    "chirps-v2.0.{year}.{month:02d}.tif.gz"
)
WORLDPOP_PAGE = "https://hub.worldpop.org/geodata/listing?id=64"
WORLDPOP_PATTERN = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/"
    "{year}/0_Mosaicked/ppp_{year}_1km_Aggregated.tif"
)
HEADERS = {"User-Agent": "FloodSense-coursework/1.0 (academic; BIC703)"}
CHIRPS_YEARS = range(1981, 2017)
POP_MIN, POP_MAX = 2000, 2020

ERA5_NOTE = """
ERA5 fallback (only if CHIRPS files are missing). Free Copernicus CDS account:
  1. Register at https://cds.climate.copernicus.eu/
  2. Save the personal access token in %USERPROFILE%\\.cdsapirc
  3. Request ERA5 monthly averaged reanalysis (single levels), variable
     total_precipitation, months 01-12, years 1981-2016.
  4. Place the files at data/external/era5/era5_total_precipitation_YYYY_MM.nc
The sampler does not invent rainfall when those files are absent.
""".strip()

GPW_NOTE = """
GPW v4 fallback (only if WorldPop files are missing). Free NASA Earthdata login:
  1. Register at https://urs.earthdata.nasa.gov/
  2. Accept the SEDAC licence at
     https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11
  3. Download GPW v4 Rev.11 population density, 30 arc-second, epochs
     2000, 2005, 2010, 2015, 2020.
  4. Place them at
     data/external/gpw/gpw_v4_population_density_rev11_YYYY_30_sec.tif
The sampler does not invent population when those files are absent.
""".strip()


def flood_table() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_CSV)
    flood = frame.loc[frame["occured"] == 1].copy()
    flood["year"] = flood["time"].astype(int)
    return flood


def coverage_report(flood: pd.DataFrame) -> dict:
    """Share of flood rows whose year (and, for CHIRPS, latitude) can be sampled."""
    year = flood["year"].to_numpy()
    lat = pd.to_numeric(flood["Latitude"], errors="coerce").to_numpy()
    n = int(len(flood))
    chirps_year = (year >= 1981) & (year <= 2016)
    chirps_domain = chirps_year & (lat >= -50) & (lat <= 50)
    pop_year = (year >= POP_MIN) & (year <= POP_MAX)
    decades = (flood["year"] // 10 * 10).value_counts().sort_index()
    return {
        "n_flood_rows": n,
        "year_min": int(year.min()),
        "year_max": int(year.max()),
        "chirps_year_share": float(chirps_year.mean()),
        "chirps_year_rows": int(chirps_year.sum()),
        "chirps_domain_share": float(chirps_domain.mean()),
        "chirps_domain_rows": int(chirps_domain.sum()),
        "worldpop_year_share": float(pop_year.mean()),
        "worldpop_year_rows": int(pop_year.sum()),
        "decades": {str(int(k)): int(v) for k, v in decades.items()},
    }


def chirps_jobs() -> list[tuple[str, Path]]:
    jobs = []
    for year in CHIRPS_YEARS:
        for month in range(1, 13):
            url = CHIRPS_PATTERN.format(year=year, month=month)
            dest = CHIRPS_DIR / f"chirps-v2.0.{year}.{month:02d}.tif.gz"
            jobs.append((url, dest))
    return jobs


def worldpop_jobs(flood: pd.DataFrame) -> list[tuple[str, Path]]:
    clipped = flood["year"].clip(POP_MIN, POP_MAX)
    years = sorted(int(year) for year in clipped.unique())
    jobs = []
    for year in years:
        url = WORLDPOP_PATTERN.format(year=year)
        dest = WORLDPOP_DIR / f"ppp_{year}_1km_Aggregated.tif"
        jobs.append((url, dest))
    return jobs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path, retries: int = 5, timeout: tuple[int, int] = (30, 120)) -> dict:
    """Download `url` to `dest`. Skip when the file already matches Content-Length."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return {"url": url, "path": dest.name, "bytes": dest.stat().st_size, "status": "skipped"}
    part = dest.with_suffix(dest.suffix + ".part")
    last_error = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            head = requests.head(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            expected = int(head.headers.get("Content-Length") or 0)
            if dest.exists() and expected and dest.stat().st_size == expected:
                return {"url": url, "path": dest.name, "bytes": expected, "status": "skipped"}
            existing = part.stat().st_size if part.exists() else 0
            headers = dict(HEADERS)
            if existing and expected and existing < expected:
                headers["Range"] = f"bytes={existing}-"
            elif existing and not expected:
                headers["Range"] = f"bytes={existing}-"
            else:
                existing = 0
            with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as response:
                if response.status_code == 416 and dest.exists():
                    return {"url": url, "path": dest.name, "bytes": dest.stat().st_size, "status": "skipped"}
                response.raise_for_status()
                if response.status_code == 200:
                    mode = "wb"
                elif response.status_code == 206:
                    mode = "ab"
                else:
                    raise RuntimeError(f"HTTP {response.status_code}")
                with part.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
            size = part.stat().st_size
            if expected and size != expected:
                raise RuntimeError(f"size {size} != {expected}")
            if size < 1000:
                raise RuntimeError(f"file is only {size} bytes")
            part.replace(dest)
            return {"url": url, "path": dest.name, "bytes": size, "status": "downloaded", "sha256": _sha256(dest)}
        except Exception as exc:  # noqa: BLE001 - retry any transport or size error
            last_error = str(exc)
            time.sleep(min(2 ** attempt, 30))
    return {"url": url, "path": dest.name, "bytes": 0, "status": "failed", "error": last_error}


def _print_manual(source: str, failed: list[dict]) -> None:
    print(f"\nMANUAL DOWNLOAD REQUIRED for {source}. {len(failed)} file(s) failed.", flush=True)
    for item in failed[:8]:
        folder = "chirps" if source == "CHIRPS" else "worldpop"
        print(f"  URL: {item['url']}", flush=True)
        print(f"  save as: data/external/{folder}/{item['path']}", flush=True)
    if len(failed) > 8:
        print(f"  ... and {len(failed) - 8} more. Names follow the same pattern.", flush=True)
    print(ERA5_NOTE if source == "CHIRPS" else GPW_NOTE, flush=True)


def _run_jobs(jobs: list[tuple[str, Path]], label: str) -> list[dict]:
    from tqdm import tqdm

    results: list[dict] = []
    lock = threading.Lock()
    bar = tqdm(total=len(jobs), desc=label, unit="file")

    def _one(job: tuple[str, Path]) -> dict:
        url, dest = job
        item = download_file(url, dest)
        with lock:
            results.append(item)
            bar.update(1)
            if item["status"] == "failed":
                bar.write(f"FAILED {dest.name}: {item.get('error')}")
            else:
                bar.write(f"{item['status']:10} {dest.name}  {item['bytes'] / 1e6:.1f} MB")
        return item

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_one, jobs))
    bar.close()
    return results


def write_manifest(coverage: dict, chirps: list[dict], worldpop: list[dict]) -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)

    def _files(rows: list[dict], folder: Path) -> list[dict]:
        out = []
        for item in sorted(rows, key=lambda row: row["path"]):
            path = folder / item["path"]
            record = {
                "name": item["path"],
                "url": item["url"],
                "bytes": int(item.get("bytes") or 0),
                "status": item["status"],
            }
            if item.get("sha256"):
                record["sha256"] = item["sha256"]
            elif path.exists() and item["status"] != "failed":
                record["sha256"] = _sha256(path)
                record["bytes"] = path.stat().st_size
            if item.get("error"):
                record["error"] = item["error"]
            out.append(record)
        return out

    payload = {
        "retrieved": date.today().isoformat(),
        "coverage": coverage,
        "sources": [
            {
                "name": "CHIRPS",
                "version": "2.0",
                "resolution": "0.05 degree",
                "years": "1981-2016 monthly (climatology uses 1981-2010)",
                "domain": "50S-50N",
                "licence": "Public domain, Climate Hazards Center, UC Santa Barbara. Cite Funk et al. 2015, Scientific Data 2:150066.",
                "url": CHIRPS_PAGE,
                "file_url_pattern": CHIRPS_PATTERN,
                "files": _files(chirps, CHIRPS_DIR),
            },
            {
                "name": "WorldPop",
                "status": "downloaded" if worldpop else "skipped",
                "version": "Global unconstrained mosaics 2000-2020, 1 km aggregated",
                "resolution": "30 arc-second (~1 km)",
                "years": f"{POP_MIN}-{POP_MAX}, nearest downloaded year; years outside that range clip to the edge",
                "units": "people per pixel",
                "licence": "CC BY 4.0. WorldPop and CIESIN (2018), doi:10.5258/SOTON/WP00647.",
                "url": WORLDPOP_PAGE,
                "file_url_pattern": WORLDPOP_PATTERN,
                "files": _files(worldpop, WORLDPOP_DIR),
            },
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST_PATH}", flush=True)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worldpop",
        action="store_true",
        help="Also download WorldPop 1 km mosaics (about 1.2 GB per year).",
    )
    args = parser.parse_args()
    if not SOURCE_CSV.exists():
        print(f"Missing {SOURCE_CSV.name}. Cannot tell which years to download.", flush=True)
        print(ERA5_NOTE, flush=True)
        print(GPW_NOTE, flush=True)
        return 2
    flood = flood_table()
    coverage = coverage_report(flood)
    print(
        f"flood rows {coverage['n_flood_rows']}  years {coverage['year_min']}-{coverage['year_max']}",
        flush=True,
    )
    print(
        f"CHIRPS year coverage {coverage['chirps_year_rows']}/{coverage['n_flood_rows']} "
        f"({coverage['chirps_year_share']:.1%}); inside 50S-50N "
        f"{coverage['chirps_domain_rows']} ({coverage['chirps_domain_share']:.1%})",
        flush=True,
    )
    print(
        f"WorldPop year coverage {coverage['worldpop_year_rows']}/{coverage['n_flood_rows']} "
        f"({coverage['worldpop_year_share']:.1%}). Earlier years clip to {POP_MIN}.",
        flush=True,
    )
    print("decade counts:", coverage["decades"], flush=True)
    chirps = _run_jobs(chirps_jobs(), "CHIRPS")
    if args.worldpop:
        worldpop = _run_jobs(worldpop_jobs(flood), "WorldPop")
    else:
        print("WorldPop skipped. Population stays missing until rasters are placed in data/external/worldpop/.", flush=True)
        print(GPW_NOTE, flush=True)
        worldpop = []
    write_manifest(coverage, chirps, worldpop)
    failed_chirps = [item for item in chirps if item["status"] == "failed"]
    failed_pop = [item for item in worldpop if item["status"] == "failed"]
    if failed_chirps:
        _print_manual("CHIRPS", failed_chirps)
    if failed_pop:
        _print_manual("WorldPop", failed_pop)
    present = any((CHIRPS_DIR / item["path"]).exists() for item in chirps) or any(
        (WORLDPOP_DIR / item["path"]).exists() for item in worldpop
    )
    if not present:
        print("No external rasters on disk. Feature build will write NaN plus missing indicators.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
