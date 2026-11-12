import shutil
from pathlib import Path

ROOT = Path("raw_data/mvtec/extracted/mvtec_anomaly_detection/mvtec_anomaly_detection")
DST = Path("dataset")

CATEGORY_TO_MATERIAL = {
    "metal_nut": "metal",
    "carpet": "fabric",
    "wood": "wood",
    "toothbrush": "plastic",
}

normal_counts = {}
defect_counts = {}

for category, material in CATEGORY_TO_MATERIAL.items():
    cat_dir = ROOT / category

    # Normal images: train/good + test/good
    normal_dir = DST / material / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        good_dir = cat_dir / split / "good"
        if not good_dir.exists():
            continue
        for img in good_dir.glob("*.png"):
            shutil.copy(img, normal_dir / f"mvtec_{category}_{split}_{img.name}")
            normal_counts[material] = normal_counts.get(material, 0) + 1

    # Defective images: test/<defect> (good ko chhodkar)
    test_dir = cat_dir / "test"
    for defect_dir in test_dir.iterdir():
        if not defect_dir.is_dir() or defect_dir.name == "good":
            continue
        out_dir = DST / material / "defective" / defect_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in defect_dir.glob("*.png"):
            shutil.copy(img, out_dir / f"mvtec_{category}_{img.name}")
            key = (material, defect_dir.name)
            defect_counts[key] = defect_counts.get(key, 0) + 1

print("NORMAL images:")
for m, n in sorted(normal_counts.items()):
    print(f"  {m}: {n}")

print("\nDEFECTIVE images:")
for (m, d), n in sorted(defect_counts.items()):
    print(f"  {m}/{d}: {n}")