import shutil
from pathlib import Path

SRC = Path("raw_data/NEU-DET")
DST = Path("dataset/metal/defective")
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

counts = {}

for split in ["train", "validation"]:
    for img_path in (SRC / split / "images").rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in EXTS:
            continue

        # Class folder ke naam se: pitted_surface/pitted_surface_229.jpg -> pitted_surface
        parent = img_path.parent.name
        if parent != "images":
            defect_class = parent.lower()
        else:
            defect_class = img_path.stem.rsplit("_", 1)[0].lower()

        out_dir = DST / defect_class
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_path, out_dir / f"{split}_{img_path.name}")
        counts[defect_class] = counts.get(defect_class, 0) + 1

print("Copied images per class:")
for name, n in sorted(counts.items()):
    print(f"  {name}: {n}")
print("Total:", sum(counts.values()))