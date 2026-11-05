#!/usr/bin/env python3
"""Convert the Roboflow COCO dataset to the YOLO format used by Ultralytics.

Source layout:

    idd.v1i.coco/
        train/_annotations.coco.json
        train/*.jpg
        valid/_annotations.coco.json
        valid/*.jpg
        test/_annotations.coco.json
        test/*.jpg

Examples
--------
Все 52 класса, исходные split'ы:

    python scripts/prepare_dataset.py --force

Только поля паспорта (как на греческом паспорте) с новым split 80/10/10:

    python scripts/prepare_dataset.py \
        --preset passport_fields --resplit \
        --output datasets/idd_passport --force

Real image copies instead of symlinks:

    python scripts/prepare_dataset.py --preset passport_fields --resplit \
        --output datasets/idd_passport --copy-images --force

The script writes images (symlinks by default), YOLO labels and ``data.yaml``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SOURCE_SPLITS = (("train", "train"), ("valid", "val"), ("test", "test"))

# Поля, которые встречаются на паспортах (греческий/сербский/латвийский и т.п.).
# Это ровно тот набор классов, что размечен на изображениях с классами photo/type.
PASSPORT_FIELDS = [
    "type",
    "code",
    "number",
    "surname",
    "surname_eng",
    "name",
    "name_eng",
    "nationality",
    "nationality-nationality_eng",
    "gender",
    "birth_date",
    "birth_place",
    "birth_place_eng",
    "birth_country",
    "issue_date",
    "expiry_date",
    "authority",
    "authority_eng",
    "authority_line0",
    "authority_line1",
    "height",
    "mrz_line0",
    "mrz_line1",
    "photo",
    "face",
]

PRESETS = {
    "all": None,
    "passport_fields": PASSPORT_FIELDS,
}

# Для пресета passport_fields берём только изображения паспортного разворота
# (у них есть рамка photo), чтобы не подмешивать другие типы документов.
PRESET_ANCHORS = {
    "passport_fields": ("photo",),
}


def coco_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    """Convert a COCO [x_min, y_min, w, h] bbox to normalized YOLO [cx, cy, w, h]."""
    x, y, w, h = bbox
    x = max(0.0, min(float(x), float(width)))
    y = max(0.0, min(float(y), float(height)))
    w = max(0.0, min(float(w), float(width) - x))
    h = max(0.0, min(float(h), float(height) - y))
    return (x + w / 2) / width, (y + h / 2) / height, w / width, h / height


def read_annotations(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_categories(source: Path) -> list[dict]:
    return read_annotations(source / "train" / "_annotations.coco.json")["categories"]


def resolve_classes(source: Path, preset: str, classes_arg: str | None) -> tuple[list[str], bool]:
    categories = sorted(get_categories(source), key=lambda c: c["id"])
    available = [c["name"] for c in categories]

    if classes_arg:
        selected = [name.strip() for name in classes_arg.split(",") if name.strip()]
    elif PRESETS[preset] is not None:
        selected = list(PRESETS[preset])
    else:
        return available, False

    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SystemExit(f"Unknown classes: {unknown}\nAvailable: {available}")

    # preserve requested order, remove duplicates
    return list(dict.fromkeys(selected)), True


def collect_samples(
    source: Path,
    selected_ids: set[int] | None,
    filter_classes: bool,
    required_ids: set[int] | None = None,
) -> dict[str, list[dict]]:
    """Load images and annotations. Returns source_split -> list of samples."""
    result: dict[str, list[dict]] = {}
    splits = list(SOURCE_SPLITS)
    if not (source / "valid").exists() and (source / "val").exists():
        splits.append(("val", "val"))

    for src_split, _ in splits:
        split_dir = source / src_split
        if not split_dir.exists():
            continue
        data = read_annotations(split_dir / "_annotations.coco.json")
        by_image: dict[int, list] = defaultdict(list)
        image_categories: dict[int, set[int]] = defaultdict(set)
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            image_categories[ann["image_id"]].add(ann.get("category_id"))
            if selected_ids is not None and ann.get("category_id") not in selected_ids:
                continue
            by_image[ann["image_id"]].append(ann)

        samples = []
        for img in data["images"]:
            if required_ids and not required_ids.issubset(image_categories.get(img["id"], set())):
                continue
            anns = by_image.get(img["id"], [])
            if filter_classes and not anns:
                continue
            path = split_dir / img["file_name"]
            if not path.exists() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            samples.append(
                {
                    "path": path,
                    "file_name": img["file_name"],
                    "width": img["width"],
                    "height": img["height"],
                    "annotations": anns,
                }
            )
        result[src_split] = samples
    return result


def write_sample(
    sample: dict,
    out_images: Path,
    out_labels: Path,
    cat_to_index: dict[int, int],
    copy_images: bool = False,
) -> bool:
    dst = out_images / sample["file_name"]
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if copy_images:
        shutil.copy2(sample["path"], dst)
    else:
        dst.symlink_to(sample["path"].resolve())

    lines = []
    for ann in sample["annotations"]:
        x, y, w, h = coco_to_yolo(ann["bbox"], sample["width"], sample["height"])
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{cat_to_index[ann['category_id']]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    (out_labels / f"{Path(sample['file_name']).stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    return True


def write_dataset(
    samples: list[dict],
    output: Path,
    cat_to_index: dict[int, int],
    split_name: str,
    copy_images: bool = False,
) -> int:
    out_images = output / "images" / split_name
    out_labels = output / "labels" / split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        write_sample(sample, out_images, out_labels, cat_to_index, copy_images)
    return len(samples)


def group_key(file_name: str) -> str:
    """Source image name without Roboflow augmentation hash.

    Example: "58_jpg.rf.42dde3....jpg" -> "58_jpg". Augmented copies of one source
    image stay in the same split, otherwise validation/test leakage inflates metrics.
    """
    return Path(file_name).stem.split(".rf.")[0]


def split_samples(samples: list[dict], val_frac: float, test_frac: float, seed: int):
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise SystemExit("val/test fractions must be >= 0 and their sum must be < 1")

    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        groups[group_key(sample["file_name"])].append(sample)

    keys = list(groups)
    random.Random(seed).shuffle(keys)

    n = len(samples)
    n_test = max(1, round(n * test_frac)) if test_frac > 0 else 0
    n_val = max(1, round(n * val_frac)) if val_frac > 0 else 0
    n_test = min(n_test, n - 2) if n >= 3 else 0
    n_val = min(n_val, n - n_test - 1) if n >= 2 else 0

    parts = {"train": [], "val": [], "test": []}
    for key in keys:
        group = groups[key]
        if len(parts["test"]) < n_test:
            parts["test"].extend(group)
        elif len(parts["val"]) < n_val:
            parts["val"].extend(group)
        else:
            parts["train"].extend(group)
    return parts


def count_classes(samples: list[dict], names_by_id: dict[int, str], cat_to_index: dict[int, int]) -> dict[str, int]:
    idx_to_name = {idx: names_by_id[cid] for cid, idx in cat_to_index.items()}
    counts = Counter()
    for sample in samples:
        for ann in sample["annotations"]:
            counts[idx_to_name[cat_to_index[ann["category_id"]]]] += 1
    return dict(counts)


def build_dataset(
    source: Path,
    output: Path,
    force: bool,
    preset: str,
    classes_arg: str | None,
    resplit: bool,
    val_frac: float,
    test_frac: float,
    seed: int,
    copy_images: bool = False,
) -> Path:
    if not (source / "train" / "_annotations.coco.json").exists():
        raise FileNotFoundError(f"Could not find a COCO dataset in {source}")

    if preset in PRESET_ANCHORS and not resplit:
        print(f"Preset '{preset}' requires a new split; enabling --resplit")
        resplit = True

    selected_names, filter_classes = resolve_classes(source, preset, classes_arg)
    categories = {c["name"]: c["id"] for c in get_categories(source)}
    selected_ids = {categories[name] for name in selected_names} if filter_classes else None
    required_ids = {categories[name] for name in PRESET_ANCHORS.get(preset, ()) if name in categories}
    cat_to_index = {categories[name]: i for i, name in enumerate(selected_names)}

    if output.exists() and force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    source_samples = collect_samples(source, selected_ids, filter_classes, required_ids)

    if resplit:
        all_samples = [sample for samples in source_samples.values() for sample in samples]
        if not all_samples:
            raise SystemExit("No images with selected classes were found")
        parts = split_samples(all_samples, val_frac, test_frac, seed)
        for split_name in ("train", "val", "test"):
            count = write_dataset(parts[split_name], output, cat_to_index, split_name, copy_images)
            print(f"{split_name:>5}: {count} images")
        stats = parts
    else:
        stats = {}
        for src_split, samples in source_samples.items():
            if not samples:
                continue
            yolo_split = "val" if src_split in {"valid", "val"} else src_split
            count = write_dataset(samples, output, cat_to_index, yolo_split, copy_images)
            stats[yolo_split] = samples
            print(f"{src_split:>5} -> {yolo_split:<5}: {count} images")

    all_written = [sample for samples in stats.values() for sample in samples]
    names_by_id = {c["id"]: c["name"] for c in get_categories(source)}
    total = count_classes(all_written, names_by_id, cat_to_index)
    print("\nAnnotations by class:")
    for name in selected_names:
        print(f"  {name:>32}: {total.get(name, 0)}")

    for split_name in ("val", "test"):
        split_counts = count_classes(stats.get(split_name, []), names_by_id, cat_to_index)
        missing = [name for name in selected_names if split_counts.get(name, 0) == 0]
        if missing:
            print(f"WARNING: classes absent in {split_name}: {missing}")

    names_yaml = "\n".join(f"  {i}: {json.dumps(name)}" for i, name in enumerate(selected_names))
    yaml_path = output / "data.yaml"
    yaml_path.write_text(
        f"# IDD document dataset (COCO -> YOLO)\n"
        f"# preset: {preset}\n"
        f"path: {output.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(selected_names)}\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )
    print(f"\nWrote {yaml_path} with {len(selected_names)} classes")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=Path("idd.v1i.coco"))
    parser.add_argument("--output", type=Path, default=Path("datasets/idd"))
    parser.add_argument("--force", action="store_true", help="Recreate the output directory")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="all",
        help="Class preset: all (52 classes) or passport_fields (поля паспорта)",
    )
    parser.add_argument("--classes", help="Comma-separated class names (overrides --preset)")
    parser.add_argument(
        "--resplit",
        action="store_true",
        help="Ignore original splits and make a new train/val/test split",
    )
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks (uses more disk space).",
    )
    args = parser.parse_args()

    build_dataset(
        source=args.source,
        output=args.output,
        force=args.force,
        preset=args.preset,
        classes_arg=args.classes,
        resplit=args.resplit,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        copy_images=args.copy_images,
    )


if __name__ == "__main__":
    main()
