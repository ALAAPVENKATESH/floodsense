"""Sample CHIRPS and population rasters at points.

Pure array helpers live here so tests do not open a network connection.
Rasterio is imported only inside `open_array` / `sample_raster_file`.

Window: a square of `window_deg` (default 0.25°) centered on the point, with an
odd pixel count so the point's own cell is the center. Longitude wraps when the
raster spans 360°. Invalid pixels are nodata, non-finite, or below `minimum`.

Population radius sum: pixels whose center is within the radius, using
111.32 km per degree and a latitude-corrected east-west length. WorldPop
pixels are people, so the sum is a sum of counts. A density raster (GPW) is
converted with the center cell's area. This is a square-window circle in local
kilometres, not a geodesic polygon.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

KM_PER_DEG = 111.32
CHIRPS_WINDOW_DEG = 0.25
POP_YEAR_MIN = 2000
POP_YEAR_MAX = 2020


def normalize_lon(lon) -> np.ndarray:
    values = np.asarray(lon, dtype=np.float64)
    return ((values + 180.0) % 360.0) - 180.0


def population_year(year: int, available, year_min: int = POP_YEAR_MIN, year_max: int = POP_YEAR_MAX):
    """Nearest downloaded epoch. `clipped` is 1 when the event year is outside the product range."""
    year = int(year)
    clipped = int(year < year_min or year > year_max)
    target = min(max(year, year_min), year_max)
    if not available:
        return None, clipped
    nearest = min((int(item) for item in available), key=lambda item: (abs(item - target), item))
    return nearest, clipped


def row_col(west, north, xsize, ysize, width, height, lon, lat):
    lon = float(np.asarray(normalize_lon(lon)).reshape(-1)[0])
    lat = float(lat)
    if abs((lon - west) - width * xsize) <= abs(xsize) * 1e-6:
        lon = float(west)
    col = int(np.floor((lon - west) / xsize))
    row = int(np.floor((north - lat) / ysize))
    if row < 0 or col < 0 or row >= height or col >= width:
        return None
    return row, col


def half_pixels(window_deg: float, cell_deg: float) -> int:
    span = max(int(round(float(window_deg) / float(cell_deg))), 1)
    if span % 2 == 0:
        span += 1
    return span // 2


def _valid(block: np.ndarray, nodata, minimum) -> np.ndarray:
    valid = np.isfinite(block)
    if nodata is not None and np.isfinite(nodata):
        valid &= block != float(nodata)
    if minimum is not None:
        valid &= block >= minimum
    return valid


def window_mean(values, row: int, col: int, half: int, nodata=None, full_globe: bool = False, minimum=None) -> float:
    arr = np.asarray(values, dtype=np.float64)
    height, width = arr.shape
    if row < 0 or col < 0 or row >= height or col >= width:
        return float("nan")
    rows = np.arange(row - half, row + half + 1)
    cols = np.arange(col - half, col + half + 1)
    rows = rows[(rows >= 0) & (rows < height)]
    if full_globe:
        cols = np.mod(cols, width)
    else:
        cols = cols[(cols >= 0) & (cols < width)]
    if len(rows) == 0 or len(cols) == 0:
        return float("nan")
    block = arr[np.ix_(rows, cols)]
    valid = _valid(block, nodata, minimum)
    if not valid.any():
        return float("nan")
    return float(block[valid].mean())


def window_means(values, rows, cols, half: int, nodata=None, full_globe: bool = False, minimum=None) -> np.ndarray:
    """Window mean at many points. Same rule as `window_mean`."""
    arr = np.asarray(values, dtype=np.float64)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    height, width = arr.shape
    acc = np.zeros(len(rows), dtype=np.float64)
    count = np.zeros(len(rows), dtype=np.float64)
    inside = (rows >= 0) & (cols >= 0) & (rows < height) & (cols < width)
    for dr in range(-int(half), int(half) + 1):
        rr = rows + dr
        for dc in range(-int(half), int(half) + 1):
            cc = cols + dc
            if full_globe:
                cc = np.mod(cc, width)
                ok = inside & (rr >= 0) & (rr < height)
            else:
                ok = inside & (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            if not np.any(ok):
                continue
            idx = np.flatnonzero(ok)
            vals = arr[rr[ok], cc[ok]]
            good = np.isfinite(vals)
            if nodata is not None and np.isfinite(nodata):
                good &= vals != float(nodata)
            if minimum is not None:
                good &= vals >= minimum
            acc[idx[good]] += vals[good]
            count[idx[good]] += 1
    out = np.full(len(rows), np.nan)
    hit = count > 0
    out[hit] = acc[hit] / count[hit]
    return out


def annual_from_months(monthly) -> tuple[float, float, float]:
    monthly = np.asarray(monthly, dtype=np.float64)
    if monthly.size == 0 or not np.isfinite(monthly).all():
        return float("nan"), float("nan"), float("nan")
    total = float(monthly.sum())
    wet = float(monthly.max())
    share = wet / total if total > 0 else float("nan")
    return total, wet, share


def standardized_anomaly(annual, history) -> float:
    history = np.asarray(history, dtype=np.float64)
    history = history[np.isfinite(history)]
    if not np.isfinite(annual) or len(history) < 2:
        return float("nan")
    std = float(history.std(ddof=1))
    if std == 0.0:
        return float("nan")
    return float((float(annual) - float(history.mean())) / std)


def cell_area_km2(lat: float, cell_deg: float, km_per_deg: float = KM_PER_DEG) -> float:
    lat_km = cell_deg * km_per_deg
    lon_km = cell_deg * km_per_deg * max(float(np.cos(np.deg2rad(lat))), 1e-6)
    return float(lat_km * lon_km)


def radius_people(values, row: int, col: int, lat: float, cell_deg: float, radius_km: float, nodata=None, full_globe: bool = False, people_per_pixel: bool = True) -> float:
    """Sum of people inside `radius_km` of the center pixel."""
    arr = np.asarray(values, dtype=np.float64)
    height, width = arr.shape
    if row < 0 or col < 0 or row >= height or col >= width:
        return float("nan")
    row_km = cell_deg * KM_PER_DEG
    col_km = cell_deg * KM_PER_DEG * max(float(np.cos(np.deg2rad(lat))), 1e-3)
    half_r = int(np.ceil(radius_km / row_km))
    half_c = int(np.ceil(radius_km / col_km))
    row_off = np.arange(-half_r, half_r + 1)
    col_off = np.arange(-half_c, half_c + 1)
    rows = row + row_off
    keep_r = (rows >= 0) & (rows < height)
    rows = rows[keep_r]
    row_off = row_off[keep_r]
    cols = col + col_off
    if full_globe:
        cols = np.mod(cols, width)
    else:
        keep_c = (cols >= 0) & (cols < width)
        cols = cols[keep_c]
        col_off = col_off[keep_c]
    if len(rows) == 0 or len(cols) == 0:
        return float("nan")
    block = arr[np.ix_(rows, cols)]
    dy = row_off.astype(np.float64) * row_km
    dx = col_off.astype(np.float64) * col_km
    inside = (dy[:, None] ** 2 + dx[None, :] ** 2) <= radius_km ** 2
    valid = _valid(block, nodata, minimum=0) & inside
    if not valid.any():
        return float("nan")
    if people_per_pixel:
        return float(block[valid].sum())
    return float(block[valid].sum() * cell_area_km2(lat, cell_deg))


def population_at_point(values, row, col, lat, cell_deg, nodata=None, full_globe: bool = False, people_per_pixel: bool = True):
    """Density (people/km²), 25 km sum, 100 km sum, and a missing flag for the center pixel."""
    if row is None or col is None:
        return float("nan"), float("nan"), float("nan"), 1
    arr = np.asarray(values, dtype=np.float64)
    center = float(arr[row, col])
    valid = bool(_valid(np.asarray([center]), nodata, minimum=0)[0])
    area = cell_area_km2(lat, cell_deg)
    if not valid or area == 0:
        density = float("nan")
    elif people_per_pixel:
        density = center / area
    else:
        density = center
    sum_25 = radius_people(arr, row, col, lat, cell_deg, 25.0, nodata, full_globe, people_per_pixel)
    sum_100 = radius_people(arr, row, col, lat, cell_deg, 100.0, nodata, full_globe, people_per_pixel)
    return density, sum_25, sum_100, 0 if valid else 1


def open_array(path) -> dict:
    """Read one band. Gzipped GeoTIFFs are uncompressed in memory."""
    import gzip

    import rasterio
    from rasterio.io import MemoryFile

    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            payload = handle.read()
        mem = MemoryFile(payload)
        src = mem.open()
    else:
        mem = None
        src = rasterio.open(path)
    try:
        arr = src.read(1)
        transform = src.transform
        nodata = src.nodata
        height, width = src.height, src.width
    finally:
        src.close()
        if mem is not None:
            mem.close()
    xsize = float(transform.a)
    ysize = float(-transform.e)
    return {
        "array": arr,
        "nodata": None if nodata is None else float(nodata),
        "west": float(transform.c),
        "north": float(transform.f),
        "xsize": xsize,
        "ysize": ysize,
        "full_globe": abs(width * xsize - 360.0) < 0.5,
    }


def sample_raster_file(path, lon: float, lat: float):
    """Center-pixel value and a missing flag. Out of bounds and nodata are (NaN, 1)."""
    grid = open_array(path)
    loc = row_col(grid["west"], grid["north"], grid["xsize"], grid["ysize"], grid["array"].shape[1], grid["array"].shape[0], lon, lat)
    if loc is None:
        return float("nan"), 1
    value = float(grid["array"][loc])
    if not bool(_valid(np.asarray([value]), grid["nodata"], minimum=None)[0]):
        return float("nan"), 1
    return value, 0
