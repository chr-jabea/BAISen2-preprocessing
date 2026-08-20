"""
Convert Sentinel-2 GeoTIFF patches into 512x512 RGB JPEGs (contrast-stretched
using the normalisation values computed by normalise.py), and update a
COCO json's image paths from .tif to .jpg accordingly.

Usage:
    # Convert the image files
    python convert_to_jpg.py \
        --input-dir DATAvX/images/ \
        --output-dir DATAvX/images_jpg/ \
        --norm-quantiles norm_quantiles.yaml

    # Update a COCO json to point to the converted .jpg files
    python convert_to_jpg.py \
        --input-json DATAvX/coco.json \
        --output-json DATAvX/coco_jpg.json
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import rasterio as rio
import yaml
from PIL import Image
from tqdm.contrib.concurrent import process_map

from normalise import normalise_image, rescale_sentinel_2


def rasterio_open(fp, normalisation_values=None, bands_to_keep=[3, 2, 1]):
    with rio.open(fp) as src:
        data = src.read()
    if data.shape[0] != 12:
        raise ValueError(
            f"{fp} doesn't have 12 bands as expected from the Sentinel-2 dataset (={data.shape[0]})"
        )
    data = data.astype(np.float32)
    data = rescale_sentinel_2(data)
    if normalisation_values is not None:
        data = normalise_image(data, normalisation_values)
    array = np.transpose(data, (1, 2, 0))
    array = np.nan_to_num(array, nan=0.0)  # just a security
    return array[:, :, bands_to_keep]


def pil_open_patch(fp, normalisation_values=None):
    """Read a Sentinel-2 GeoTIFF patch as an 8-bit RGB PIL image; falls back
    to a plain PIL open for any other file type."""
    fp = Path(fp)
    if fp.suffix.lower() in (".tif", ".tiff"):
        arr = rasterio_open(fp, normalisation_values)
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    return Image.open(fp)


def convert_image(f, input_dir, output_dir, normalisation_values):
    img = pil_open_patch(f, normalisation_values)
    out_p = output_dir / f.relative_to(input_dir).with_suffix(".jpg")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_p, quality=95, subsampling=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Sentinel-2 GeoTIFF patches to JPEG"
    )
    parser.add_argument(
        "--input-dir", type=str, default=None, help="input directory path"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="output directory path"
    )
    parser.add_argument(
        "--norm-quantiles",
        type=str,
        default=None,
        help="yaml file with the per-band [min, max] normalisation values (see normalise.py)",
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="coco dataset json with .tif image paths",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="coco dataset json with .jpg image paths",
    )
    args = parser.parse_args()

    if (args.input_dir is not None) and (args.output_dir is not None):
        if not args.norm_quantiles:
            parser.error("--norm-quantiles is required when converting images")
        with open(args.norm_quantiles) as f:
            normalisation_values = yaml.safe_load(f)
        normalisation_values = {
            k: [v[0] / 10000.0, v[1] / 10000.0] for k, v in normalisation_values.items()
        }

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        files = list(input_dir.glob("**/*.tif"))
        process_map(
            partial(
                convert_image,
                input_dir=input_dir,
                output_dir=output_dir,
                normalisation_values=normalisation_values,
            ),
            files,
            max_workers=8,
        )

    if (args.input_json is not None) and (args.output_json is not None):
        with open(args.input_json) as f:
            coco_data = json.load(f)

        for image in coco_data.get("images", []):
            file_path = Path(image["file_name"])
            if file_path.suffix.lower() == ".tif":
                image["file_name"] = str(file_path.with_suffix(".jpg"))

        with open(args.output_json, "w") as f:
            json.dump(coco_data, f, indent=4)
