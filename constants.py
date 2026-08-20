"""Shared constants for the preprocessing pipeline."""

import numpy as np

FILL_VALUE = np.uint16(65535)  # 0 cannot be used (valid reflectance value)

# Sentinel-2 L2A bands used to build the image patches
BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]
