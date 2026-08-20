"""
Split a COCO dataset into train/test so that no two geographically
overlapping image patches end up in different splits

Usage:
    python split_dataset.py \
        --input-json DATA/coco.json \
        --output-dir DATA/splits/ \
        --train-ratio 0.7 \
        --test-ratio 0.3
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

from shapely.geometry import box

sys.setrecursionlimit(5000)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split_coco_json_clustered_unionfind(
    input_json_path,
    output_dir,
    train_ratio=0.7,
    test_ratio=0.3,
    overlap_thresh=0.0,
    seed=42,
):
    random.seed(seed)

    with open(input_json_path) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    beaches = coco.get("beaches", [])

    for img in images:
        xmin, ymin, xmax, ymax = img["bounds"]
        img["geometry"] = box(xmin, ymin, xmax, ymax)

    parent = {img["id"]: img["id"] for img in images}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            geom1, geom2 = images[i]["geometry"], images[j]["geometry"]
            inter_area = geom1.intersection(geom2).area
            if (
                inter_area / geom1.area > overlap_thresh
                or inter_area / geom2.area > overlap_thresh
            ):
                union(images[i]["id"], images[j]["id"])

    clusters_dict = defaultdict(list)
    for img in images:
        clusters_dict[find(img["id"])].append(img)
    clusters = list(clusters_dict.values())
    logger.info("%d geographic clusters", len(clusters))

    random.shuffle(clusters)
    total_ann = sum(
        len([a for a in annotations if a["image_id"] in [img["id"] for img in cluster]])
        for cluster in clusters
    )
    n_train = int(train_ratio * total_ann)
    n_test = total_ann - n_train

    splits = {"train": [], "test": []}
    split_counts = {"train": 0, "test": 0}

    for cluster in clusters:
        cluster_img_ids = [img["id"] for img in cluster]
        cluster_ann_count = len(
            [a for a in annotations if a["image_id"] in cluster_img_ids]
        )
        deficits = {
            k: ([n_train, n_test][i] - split_counts[k]) / [n_train, n_test][i]
            for i, k in enumerate(["train", "test"])
        }
        split_name = max(deficits, key=deficits.get)
        splits[split_name].append(cluster)
        split_counts[split_name] += cluster_ann_count

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for split_name, cluster_list in splits.items():
        split_imgs = [img for cluster in cluster_list for img in cluster]
        img_ids = {img["id"] for img in split_imgs}
        split_anns = [ann for ann in annotations if ann["image_id"] in img_ids]
        split_beaches_ids = {img["beach"] for img in split_imgs}
        split_beaches = [b for b in beaches if b["id"] in split_beaches_ids]

        for img in split_imgs:
            del img["geometry"]

        out = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": split_imgs,
            "annotations": split_anns,
            "beaches": split_beaches,
            "categories": coco.get("categories", []),
        }
        out_path = Path(output_dir) / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(
            "%s: %d images, %d annotations",
            split_name,
            len(split_imgs),
            len(split_anns),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a COCO dataset into train/test with geographically disjoint clusters"
    )
    parser.add_argument("--input-json", required=True, help="input COCO json file")
    parser.add_argument(
        "--output-dir", required=True, help="output directory for train.json/test.json"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="target fraction of annotations in train",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.3,
        help="target fraction of annotations in test",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="random seed for the cluster shuffle"
    )
    args = parser.parse_args()

    split_coco_json_clustered_unionfind(
        input_json_path=args.input_json,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
