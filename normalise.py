"""
Compute the 2%-98% percentile normalisation values (per band) over a
directory of Sentinel-2 image patches, using histograms for efficiency.

Usage:
    python normalise.py \
        --images-dir DATAvX/images/ \
        --dst norm_quantiles.yaml
"""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
import rasterio as rio

from constants import FILL_VALUE, BANDS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def rescale_sentinel_2(array):
    return array / 10000.0


def get_hist_norm_values(input_folder_path, output_file=None, nbins=65535):
    """
    Using histograms for efficiency, useful for floats.
    """
    variables = BANDS
    input_folder = Path(input_folder_path)
    img_paths = list(input_folder.glob("*/*.tif"))
    bin_edges = np.linspace(0, 65535, nbins + 1)
    histograms = {var: np.zeros(nbins, dtype=np.float64) for var in variables}

    for path in tqdm(img_paths):
        with rio.open(path) as img:
            for idx_v, variable in enumerate(variables):
                band = img.read(idx_v + 1).flatten()
                band = band[band != FILL_VALUE]
                if band.size == 0:
                    continue
                hist, _ = np.histogram(band, bins=bin_edges)
                histograms[variable] += hist
    if output_file is None:
        return histograms
    else:
        np.savez(output_file, **histograms, bin_edges=bin_edges)


def load_hists_from_npz(npz_file):
    data = np.load(npz_file)
    hists = {}
    for key in data.files:
        if key != "bin_edges":
            hists[key] = data[key]
    bin_edges = data["bin_edges"]
    return hists, bin_edges


def get_normalisation_values(files):
    variables = BANDS
    bin_edges = load_hists_from_npz(files[0])[1]
    hists_list = [load_hists_from_npz(file)[0] for file in files]
    quantiles = {}
    hists_sum = {k: np.zeros_like(v) for k, v in hists_list[0].items()}
    for hists in hists_list:
        for variable in variables:
            hists_sum[variable] += hists[variable]

    for variable in variables:
        hist = hists_sum[variable]
        cumsum = np.cumsum(hist)
        total = cumsum[-1]
        if total == 0:
            quantiles[variable] = [np.nan, np.nan]
            continue
        q2 = np.searchsorted(cumsum, 0.02 * total)
        q98 = np.searchsorted(cumsum, 0.98 * total)
        quantiles[variable] = [bin_edges[q2], bin_edges[q98]]

    return quantiles


def normalise_image(image, normalisation_values):
    for idx, (key, value) in enumerate(normalisation_values.items()):
        Xmin, Xmax = value
        image[idx, ...] = np.where(image[idx, ...] > Xmax, Xmax, image[idx, ...])
        image[idx, ...] = np.where(image[idx, ...] < Xmin, Xmin, image[idx, ...])
        image[idx, ...] = (image[idx, ...] - Xmin) / (Xmax - Xmin)

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the 2%-98% percentile normalisation values (per band) "
        "over a directory of Sentinel-2 image patches"
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="path to the image patches directory (train split only)",
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="destination yaml file for the normalisation values",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        hist_file = Path(tmp_dir) / "histograms.npz"
        get_hist_norm_values(args.images_dir, output_file=hist_file)
        quantiles = get_normalisation_values([hist_file])

    with open(args.dst, "w") as f:
        yaml.safe_dump({k: [float(v[0]), float(v[1])] for k, v in quantiles.items()}, f)
    logger.info("Wrote normalisation values to %s", args.dst)
