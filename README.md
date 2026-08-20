# BAISen2 dataset — preprocessing code
Code used to preprocess Sentinel-2 imagery and rip-channel annotations
into the BAISen2 dataset released alongside the data paper.

## Setup
```
pip install -r requirements.txt
```
## Pipeline
1. **`transform_dataset.py`** — cut annotated scenes into fixed-size
   patches and download the matching Sentinel-2 imagery from the Planetary
   Computer.
2. **`build_coco_dataset.py`** — build a COCO dataset.
3. **`split_dataset.py`** — split the COCO dataset into train/test with
   geographically disjoint clusters.
4. **`normalise.py`** — compute 2%-98% percentile normalisation values
   from the train images.
5. **`convert_to_jpg.py`** — convert the patches to JPEG

## Notes
- **YOLO format.** The YOLO version of the dataset was generated from the
  COCO json with `ultralytics.data.converter.convert_coco()`
  (`pip install ultralytics`). Images without annotations need an empty
  `.txt` label file, which `convert_coco()` does not create automatically.
- **Temporal split.** In addition to the spatial split (`split_dataset.py`), the paper
  also reports a temporal split: images from 2015-2021 → train, images from
  2022-2024 → test, based on each image's acquisition year (`date_captured`
  field / `YYYY` filename prefix).


## License
Code released under the MIT License. The dataset itself is released under
CC BY 4.0.

## Citation
If you use this code or the BAISen2 dataset, please cite:
```bibtex
@article{baisen2,
  title   = {A Sentinel-2 image dataset for rip channels detection along the French Atlantic coast},
  author  = {Jabea, Christopher and Dantas, C{\'a}ssio F. and Castelle, Bruno and Manighetti, Isabelle and Ienco, Dino},
  journal = {},
  year    = {2026},
  note    = {Under review}
}
```
