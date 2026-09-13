import json
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

from pycocotools.coco import COCO

DATASET_ROOT = Path(__file__).parent / "dataset"
ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_BASE_URL = {
    "train2017": "http://images.cocodataset.org/train2017/",
    "val2017": "http://images.cocodataset.org/val2017/",
}

# accessibility-relevant classes: people, vehicles, and common obstacles a
# navigation-assist app would want to call out.
TARGET_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "fire hydrant", "stop sign", "bench",
    "backpack", "suitcase", "dog", "chair", "potted plant",
]

MAX_IMAGES_PER_SPLIT = {
    "train2017": 3000,
    "val2017": 500,
}

DOWNLOAD_WORKERS = 16


def download_annotations():
    zip_path = DATASET_ROOT / "annotations_trainval2017.zip"
    ann_dir = DATASET_ROOT / "annotations"
    needed = [f"annotations/instances_{split}.json" for split in MAX_IMAGES_PER_SPLIT]
    if not all((DATASET_ROOT / n).exists() for n in needed):
        if not zip_path.exists():
            print(f"Downloading annotations from {ANNOTATIONS_ZIP_URL} ...")
            urlretrieve(ANNOTATIONS_ZIP_URL, zip_path)
        print("Extracting instances_*.json only (skipping captions/keypoints)...")
        with zipfile.ZipFile(zip_path) as zf:
            for member in needed:
                zf.extract(member, DATASET_ROOT)
        zip_path.unlink()  # keep only the extracted JSON, drop the zip
    return ann_dir


def _download_one(url, dest):
    if not dest.exists():
        urlretrieve(url, dest)


def filter_and_download(split, ann_dir, max_images):
    ann_file = ann_dir / f"instances_{split}.json"
    coco = COCO(str(ann_file))

    cat_ids = coco.getCatIds(catNms=TARGET_CLASSES)
    found_names = {c["name"] for c in coco.loadCats(cat_ids)}
    missing = set(TARGET_CLASSES) - found_names
    if missing:
        print(f"Warning: classes not found in COCO: {missing}")

    # Per-category image-id sets (still one call per category -- getImgIds
    # with a multi-item catIds list is an AND filter, not OR).
    per_cat_img_ids = {
        cat_id: set(coco.getImgIds(catIds=[cat_id])) for cat_id in cat_ids
    }

    quota_per_class = max_images // len(cat_ids)
    selected_ids = set()
    for cat_id in sorted(cat_ids, key=lambda c: len(per_cat_img_ids[c])):
        remaining_budget = max_images - len(selected_ids)
        if remaining_budget <= 0:
            break
        take = min(quota_per_class, remaining_budget)
        new_ids = sorted(per_cat_img_ids[cat_id] - selected_ids)[:take]
        selected_ids.update(new_ids)
    img_ids = sorted(selected_ids)
    print(
        f"{split}: {len(img_ids)} images matched "
        f"(capped at {max_images}, balanced rarest-class-first, "
        f"quota {quota_per_class}/class)"
    )

    img_dir = DATASET_ROOT / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)

    imgs = coco.loadImgs(img_ids)
    jobs = [(IMAGE_BASE_URL[split] + img["file_name"], img_dir / img["file_name"]) for img in imgs]

    done = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = [pool.submit(_download_one, url, dest) for url, dest in jobs]
        for _ in as_completed(futures):
            done += 1
            if done % 200 == 0:
                print(f"  {split}: {done}/{len(jobs)} images downloaded")

    # write a filtloadAnnsered annotation file containing only the kept images/cats
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=cat_ids)
    anns = coco.loadAnns(ann_ids)
    filtered = {
        "images": imgs,
        "annotations": anns,
        "categories": coco.loadCats(cat_ids),
    }
    out_path = DATASET_ROOT / f"instances_{split}_subset.json"
    with open(out_path, "w") as f:
        json.dump(filtered, f)
    print(f"{split}: wrote {out_path} ({len(anns)} annotations, {len(imgs)} images)")

    cat_names = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}
    counts = Counter(a["category_id"] for a in anns)
    print(f"{split}: per-class annotation counts:")
    for cat_id, count in counts.most_common():
        print(f"  {cat_names[cat_id]:15s} {count}")

    kept_filenames = {img["file_name"] for img in imgs}
    stale = [p for p in img_dir.iterdir() if p.name not in kept_filenames]
    for path in stale:
        path.unlink()
    if stale:
        print(f"{split}: removed {len(stale)} stale images no longer in the selection")

    # done with the full ~450MB/20MB annotation file, keep only our small subset
    ann_file.unlink()


if __name__ == "__main__":
    ann_dir = download_annotations()
    for split, max_images in MAX_IMAGES_PER_SPLIT.items():
        filter_and_download(split, ann_dir, max_images)
