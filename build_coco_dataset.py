"""
Convert a rip-channel dataset — GeoTIFF images with one GeoJSON (MultiPolygon) 
annotation file per image — into a single COCO-format dataset.

Expected input layout and filename convention:
    images_dir/<area>/<YYYY-MM-DD>-<area>-s2-rgb[-p<N>].tif
    annotations_dir/<area>/<YYYY-MM-DD>-<area>-s2-rgb-w<XX>[-p<N>].geojson

Usage:
    python build_coco_dataset.py \
        --images-dir DATA/images/ \
        --annotations-dir DATA/annotations/ \
        --dst DATA/coco.json
"""

import json
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime

import rasterio as rio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_coco_json():
    """Initialise an empty COCO dataset."""
    return {
        "info": {
            "years": datetime.now().year,
            "version": 1.0,
            "description": "BAISen2 object detection dataset",
            "contributor": "",
            "url": "",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [
            {
                "id": 0,
                "name": "Copernicus Sentinel-2",
                "url": "https://www.copernicus.eu/en/terms-use/how-access-data",
            }
        ],
        "beaches": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "baine", "supercategory": ""}],
    }


def is_empty_multipolygon(geometry):
    coords = geometry.get("coordinates", [])
    if not coords:
        return True
    for polygon in coords:
        for ring in polygon:
            if len(ring) > 0:
                return False
    return True


def get_multipolygon_bounds(multipolygon):
    coords = multipolygon["coordinates"][0][0]
    xmin = min(p[0] for p in coords)
    ymin = min(p[1] for p in coords)
    xmax = max(p[0] for p in coords)
    ymax = max(p[1] for p in coords)
    return [xmin, ymin, xmax, ymax]


def coords_to_pixels(img_bounds, img_width_px, img_height_px, ann_bounds):
    img_l, img_b, img_r, img_t = img_bounds  # [left, bottom, right, top]
    ann_l, ann_b, ann_r, ann_t = ann_bounds
    img_w, img_h = img_r - img_l, img_t - img_b  # width and height

    ann_pl = round((ann_l - img_l) / img_w * img_width_px)
    ann_pt = round((img_t - ann_t) / img_h * img_height_px)
    ann_pw = round((ann_r - ann_l) / img_w * img_width_px)
    ann_ph = round((ann_t - ann_b) / img_h * img_height_px)
    return [ann_pl, ann_pt, ann_pw, ann_ph]


def convert_geojson_to_coco(geojson_path, img_info, ann_id):
    """Convert one GeoJSON annotation file into COCO-format annotations."""
    with open(geojson_path) as f:
        data = json.load(f)

    coco_annotations = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry")
        if geometry.get("type") != "MultiPolygon":
            raise ValueError(
                f"MultiPolygon type was expected, found {geometry.get('type')}"
            )
        if is_empty_multipolygon(geometry):
            continue

        ann_bounds = get_multipolygon_bounds(geometry)
        pixels_bounds = coords_to_pixels(
            img_info["bounds"], img_info["width"], img_info["height"], ann_bounds
        )
        area = pixels_bounds[2] * pixels_bounds[3]
        ann_id += 1

        coco_annotations.append(
            {
                "id": ann_id,
                "image_id": img_info["id"],
                "category_id": 0,
                "bbox": pixels_bounds,
                "area": area,
                "iscrowd": 0,
            }
        )

    logger.info("%d annotations added for %s", len(coco_annotations), geojson_path)
    return coco_annotations, ann_id


def build_coco_dataset(images_dir, annotations_dir, dst):
    """Build a single COCO-format dataset from the images and annotations."""
    coco_data = init_coco_json()

    beach_to_index = {}
    images = []
    short_paths = sorted(
        (Path(*p.parts[-2:]) for p in images_dir.rglob("*.tif")),
        key=lambda p: (p.parts[0], p.parts[1]),
    )

    for idx, path in enumerate(short_paths):
        match = re.search(r"(.*?)/.*-(.*?)-s2-rgb.*\.tif$", str(path))
        if not match:
            logger.warning("%s path not well formatted", path)
            continue

        beach_name = match.group(2).lower()
        with rio.open(images_dir / path) as src:
            bounds = src.bounds
            res_x, res_y = src.res
            width = round((bounds.right - bounds.left) / res_x)
            height = round((bounds.top - bounds.bottom) / res_y)
            crs = src.crs

        if beach_name not in beach_to_index:
            beach_id = len(beach_to_index)
            beach_to_index[beach_name] = beach_id
            coco_data["beaches"].append({"id": beach_id, "name": beach_name})

        images.append(
            {
                "id": idx,
                "width": width,
                "height": height,
                "file_name": path.as_posix(),
                "crs": "EPSG:" + str(crs.to_epsg()),
                "date_captured": "-".join(path.name.split("-")[:3]),
                "license": 0,
                "bounds": list(bounds),
                "beach": beach_to_index[beach_name],
                "flickr_url": "",
                "coco_url": "",
            }
        )
    coco_data["images"] = images

    images_info = {img["file_name"]: img for img in coco_data["images"]}
    ann_id = 0
    annotations = []
    for beach_folder in (d for d in annotations_dir.iterdir() if d.is_dir()):
        for path in beach_folder.glob("*.geojson"):
            filename = path.stem
            file_beach_name = filename.split("-")[3]

            match = re.match(r"^(.*)-w\d+(-p\d+)", filename)
            if not match:
                logger.error("%s does not follow the expected filename format", path)
                continue
            image_key = f"{file_beach_name}/{match.group(1)}{match.group(2)}.tif"
            img_info = images_info.get(image_key)
            if img_info is None:
                logger.error("%s is not a valid image", image_key)
                continue

            ann, ann_id = convert_geojson_to_coco(path, img_info, ann_id)
            annotations += ann

    coco_data["annotations"] = annotations

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=4, ensure_ascii=False)
    logger.info(
        "Wrote %d images and %d annotations to %s", len(images), len(annotations), dst
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a GeoTIFF + GeoJSON rip-channel dataset to the COCO format"
    )
    parser.add_argument(
        "--images-dir", required=True, help="path to the images directory"
    )
    parser.add_argument(
        "--annotations-dir", required=True, help="path to the annotations directory"
    )
    parser.add_argument("--dst", required=True, help="destination COCO json file")
    args = parser.parse_args()

    build_coco_dataset(
        Path(args.images_dir),
        Path(args.annotations_dir),
        Path(args.dst),
    )
