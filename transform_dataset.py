"""
Cut annotated Sentinel-2 scenes into fixed-size patches with minimal overlap
between annotations (greedy set-cover over randomly placed candidate
windows)
Optionnaly download the fixed-sized images

Usage:
    python transform_dataset.py \
        --annotations-dir DATA/annotations/ \
        --images-dir DATA/images/ \
        --global-annotations-dir DATA/global_anns/ \
        --img-output-dir DATAvX/images/ \
        --ann-output-dir DATAvX/annotations/ \
        --download
"""

import sys
import argparse
import logging
import json
import random
import re
from pathlib import Path

import numpy as np
from shapely.geometry import shape, box, mapping, Polygon, MultiPolygon
from pyproj import Transformer
import rasterio as rio
import xarray as xr
import rioxarray
import pystac_client
import planetary_computer
import stackstac
from dask import delayed, compute
from dask.diagnostics import ProgressBar

from constants import FILL_VALUE, BANDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 1000


def is_empty_multipolygon(geometry):
    coords = geometry.get("coordinates", [])
    if not coords:
        return True
    for polygon in coords:
        for ring in polygon:
            if len(ring) > 0:
                return False
    return True


def extract_bboxes(path):
    """Load the (non-empty) MultiPolygon annotations of a GeoJSON file."""
    with open(path) as f:
        data = json.load(f)
    crs = data.get("crs", {}).get("properties", {}).get("name")
    features = data.get("features", [])
    if not features:
        return None, None

    bboxes = [
        shape(f["geometry"])
        for f in features
        if not is_empty_multipolygon(f.get("geometry"))
    ]
    return bboxes, crs


def centered_random_patch_from_image(image_path, size, resolution=10, max_shift=20):
    """Build one candidate window centered (with a random shift) on an image
    that has no annotation, so it still gets included in the dataset."""
    with rio.open(image_path) as src:
        bounds = src.bounds  # (xmin, ymin, xmax, ymax)
        crs = src.crs

    offset = size // 2 * resolution
    max_shift = max_shift * offset / 100
    dx = random.uniform(-max_shift, max_shift)
    dy = random.uniform(-max_shift, max_shift)
    cx = (bounds.left + bounds.right) / 2 + dx
    cy = (bounds.bottom + bounds.top) / 2 + dy
    xmin = cx - offset
    xmax = xmin + (size - 1) * resolution
    ymin = cy - offset
    ymax = ymin + (size - 1) * resolution

    return np.array([[xmin, ymin, xmax, ymax]]), crs


def create_random_extents(bboxes, size, resolution=10, max_shift=20):
    """Build one candidate window per annotation, centered (with a random
    shift) on that annotation, and record which annotations end up fully
    inside / partially overlapping each candidate window."""
    offset = size // 2 * resolution
    max_shift = max_shift * offset / 100
    extents, contains, overlaps = [], [], []

    for geometry in bboxes:
        centroid = geometry.centroid
        dx = random.uniform(-max_shift, max_shift)
        dy = random.uniform(-max_shift, max_shift)
        cx, cy = centroid.x + dx, centroid.y + dy
        xmin = cx - offset
        xmax = xmin + (size - 1) * resolution
        ymin = cy - offset
        ymax = ymin + (size - 1) * resolution
        bbox = [xmin, ymin, xmax, ymax]
        container = box(*bbox)

        extents.append(bbox)
        contains.append([container.contains(g) for g in bboxes])
        overlaps.append(
            [g.intersects(container) and not container.contains(g) for g in bboxes]
        )

    return np.array(extents), np.array(contains), np.array(overlaps)


def clipped_or_inside(a, b):
    """Return a if it's entirely inside b, otherwise the intersection."""
    return a if b.contains(a) else a.intersection(b)


def intersection_ratio(a, b):
    """Fraction of `a`'s area that falls inside `b` (between 0 and 1)."""
    if a.area < 1e-9 or not a.intersects(b):
        return 0.0
    return a.intersection(b).area / a.area


def get_inside_bboxes(bbox, bboxes):
    """Return the parts of bboxes that fall at least 50% inside bbox,
    clipped to bbox when they're not entirely inside it."""
    if bboxes is None:
        return []
    container = box(*bbox)
    return [
        clipped_or_inside(poly, container)
        for poly in bboxes
        if intersection_ratio(poly, container) >= 0.5
    ]


def filter_overlaps(extents, contains, overlaps):
    kept = np.where(~np.any(overlaps, axis=1))[0]
    return extents[kept], contains[kept]


def greedy_max_coverage(extents, contains):
    """Greedily select the minimal set of candidate windows that together
    cover every annotation at least once."""
    anns_in_windows = [set(np.where(row)[0]) for row in contains]
    all_anns = set(range(len(contains[0])))
    selected_windows, selected_windows_annotations = [], []
    covered_annotations = set()

    while covered_annotations != all_anns:
        best_cover, best_idx = set(), -1
        for i, ann in enumerate(anns_in_windows):
            new_cover = ann - covered_annotations
            if len(new_cover) > len(best_cover):
                best_cover, best_idx = new_cover, i
        if best_idx == -1:
            missing_anns = len(all_anns - covered_annotations)
            return extents[selected_windows], selected_windows_annotations, missing_anns

        selected_windows.append(best_idx)
        selected_windows_annotations.append(anns_in_windows[best_idx])
        covered_annotations.update(best_cover)

    return extents[selected_windows], selected_windows_annotations, 0


def is_list_deep_empty(lst):
    if not lst:
        return True
    for item in lst:
        if isinstance(item, list):
            if not is_list_deep_empty(item):
                return False
        else:
            return False
    return True


def shapely_list_to_geojson(shapes, crs_epsg, out_file):
    """Convert a list of Polygons/MultiPolygons into a GeoJSON FeatureCollection."""
    features = []
    if not is_list_deep_empty(shapes):
        for i, geom in enumerate(shapes, 1):
            if isinstance(geom, Polygon):
                geom = MultiPolygon([geom])
            features.append(
                {"type": "Feature", "properties": {"id": i}, "geometry": mapping(geom)}
            )

    geojson_dict = {
        "type": "FeatureCollection",
        "name": Path(out_file).stem,
        "crs": {"type": "name", "properties": {"name": str(crs_epsg)}},
        "features": features,
    }
    with open(out_file, "w") as f:
        json.dump(geojson_dict, f, indent=4)


def build_patch_windows(annotations_dir, images_dir, global_ann_dir, size):
    """For each annotation file, select the minimal set of size x size
    windows that together cover all its annotations"""
    date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
    annotations = {}
    n_all_anns, n_missing_anns = 0, 0

    for beach_folder in (d for d in annotations_dir.iterdir() if d.is_dir()):
        for path in beach_folder.glob("*.geojson"):
            bboxes, crs = extract_bboxes(path)
            date_match = date_pattern.search(path.name)
            if not date_match:
                raise ValueError(f"Wrong filename format: {path}")
            global_ann_path = global_ann_dir / (date_match.group() + ".geojson")
            global_bboxes, _ = extract_bboxes(global_ann_path)

            if bboxes is None:
                logger.warning(
                    "No features found in %s, adding the image without annotations",
                    path,
                )
                img_path = images_dir / re.sub(
                    r"-w\d{2}\.geojson$", ".tif", str(path.relative_to(annotations_dir))
                )
                win, crs = centered_random_patch_from_image(img_path, size)
                epsg_code = crs.to_epsg()
                if epsg_code is None:
                    raise ValueError(f"{img_path} has no EPSG code")
                annotations[path] = (
                    win,
                    [get_inside_bboxes(win[0], global_bboxes)],
                    f"urn:ogc:def:crs:EPSG::{epsg_code}",
                )
                continue

            n_all_anns += len(bboxes)
            for attempt in range(MAX_ATTEMPTS):
                extents, contains, overlaps = create_random_extents(bboxes, size)
                f_extents, f_contains = filter_overlaps(extents, contains, overlaps)
                if f_contains.size == 0:
                    if attempt != MAX_ATTEMPTS - 1:
                        continue
                    logger.warning("Missing annotations in %s", path)
                    n_missing_anns += len(bboxes)
                    break

                selected_windows, _, missing_anns = greedy_max_coverage(
                    f_extents, f_contains
                )
                if missing_anns:
                    if attempt == MAX_ATTEMPTS - 1:
                        n_missing_anns += missing_anns
                        logger.warning("Missing annotations in %s", path)
                else:
                    selected_bboxes_list = [
                        get_inside_bboxes(win, global_bboxes)
                        for win in selected_windows
                    ]
                    annotations[path] = (selected_windows, selected_bboxes_list, crs)
                    break

    logger.info("n_missing_anns=%d n_all_anns=%d", n_missing_anns, n_all_anns)
    if n_missing_anns:
        raise RuntimeError(
            f"{n_missing_anns} annotations could not be placed in a window, try again"
        )
    return annotations


def process_window(
    path,
    idx,
    w,
    ann,
    epsg_code,
    date,
    bands,
    catalog,
    fill_value,
    img_output_dir,
    ann_output_dir,
):
    """Download the Sentinel-2 imagery for one window and save the patch +
    its annotations."""
    beach_name = path.parent.name

    ann_output_dir = Path(ann_output_dir) / beach_name
    ann_output_dir.mkdir(parents=True, exist_ok=True)
    shapely_list_to_geojson(
        ann,
        f"urn:ogc:def:crs:EPSG::{epsg_code}",
        ann_output_dir / f"{path.stem}-p{idx}.geojson",
    )

    xmin, ymin, xmax, ymax = w
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    xmin_wgs84, ymin_wgs84 = transformer.transform(xmin, ymin)
    xmax_wgs84, ymax_wgs84 = transformer.transform(xmax, ymax)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=[xmin_wgs84, ymin_wgs84, xmax_wgs84, ymax_wgs84],
        datetime=date,
    )
    items = search.item_collection()

    stack = stackstac.stack(
        items,
        assets=bands,
        epsg=epsg_code,
        resolution=10,
        bounds=list(w),
        rescale=False,
        dtype="uint16",
        fill_value=fill_value,
    )
    if len(stack.time) > 1:
        valid = stack != fill_value
        sum_val = stack.where(valid, 0).sum(dim="time")
        count = valid.sum(dim="time")
        stack = xr.where(count > 0, sum_val / count, fill_value).astype("uint16")
    if "time" in stack.dims:
        stack = stack.squeeze("time", drop=True)

    img_output_dir = Path(img_output_dir) / beach_name
    img_output_dir.mkdir(parents=True, exist_ok=True)
    match = re.match(r"^(.*?-rgb)", path.stem)
    if not match:
        raise ValueError(f"Wrong filename format {path}")
    stack.rio.to_raster(img_output_dir / f"{match.group(1)}-p{idx}.tif")


def download_patches(annotations, bands, fill_value, img_output_dir, ann_output_dir):
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    tasks, total_ann = [], 0
    for path, (windows, anns, crs) in annotations.items():
        epsg_match = re.search(r"EPSG::(\d+)", crs)
        if not epsg_match:
            raise ValueError(f"{path} does not contain a valid CRS")
        epsg_code = int(epsg_match.group(1))

        date_match = re.search(r"\d{4}-\d{2}-\d{2}", path.stem)
        if not date_match:
            raise ValueError(f"{path} does not follow the expected filename format")
        date = date_match.group()

        for idx, (w, ann) in enumerate(zip(windows, anns)):
            if not is_list_deep_empty(ann):
                total_ann += len(ann)
            tasks.append(
                delayed(process_window)(
                    path,
                    idx,
                    w,
                    ann,
                    epsg_code,
                    date,
                    bands,
                    catalog,
                    fill_value,
                    img_output_dir,
                    ann_output_dir,
                )
            )

    with ProgressBar():
        compute(*tasks, scheduler="threads", num_workers=1)
    logger.info("Downloaded %d patches (%d annotations)", len(tasks), total_ann)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut annotated scenes into fixed-size patches and optionally download the matching imagery from the Planetary Computer"
    )
    parser.add_argument(
        "--annotations-dir",
        required=True,
        help="path to the per-beach annotations directory",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="path to the source images",
    )
    parser.add_argument(
        "--global-annotations-dir",
        required=True,
        help="path to the per-date global annotations directory",
    )
    parser.add_argument(
        "--img-output-dir", required=True, help="output directory for the image patches"
    )
    parser.add_argument(
        "--ann-output-dir",
        required=True,
        help="output directory for the patch annotations",
    )
    parser.add_argument(
        "--size", "-s", type=int, default=512, help="patch size in pixels"
    )
    parser.add_argument(
        "--download",
        "-d",
        action="store_true",
        help="download the imagery patches from the Planetary Computer",
    )
    args = parser.parse_args()

    annotations = build_patch_windows(
        Path(args.annotations_dir),
        Path(args.images_dir),
        Path(args.global_annotations_dir),
        args.size,
    )

    if args.download:
        download_patches(
            annotations,
            BANDS,
            FILL_VALUE,
            Path(args.img_output_dir),
            Path(args.ann_output_dir),
        )
