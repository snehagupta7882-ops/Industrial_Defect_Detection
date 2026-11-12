from pathlib import Path
from PIL import Image

ROOT = Path("dataset")
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

counts = {}
corrupt = []
sizes = {}
modes = {}

for img_path in ROOT.rglob("*"):
    if not img_path.is_file() or img_path.suffix.lower() not in EXTS:
        continue

    rel = img_path.relative_to(ROOT).parts          # (material, condition, [defect], file)
    key = "/".join(rel[:-1])
    counts[key] = counts.get(key, 0) + 1

    try:
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im:
            sizes[im.size] = sizes.get(im.size, 0) + 1
            modes[im.mode] = modes.get(im.mode, 0) + 1
    except Exception as e:
        corrupt.append((str(img_path), str(e)))

print("IMAGES PER FOLDER")
for k, n in sorted(counts.items()):
    print(f"  {k}: {n}")
print("\nTOTAL:", sum(counts.values()))

print("\nTOP IMAGE SIZES")
for s, n in sorted(sizes.items(), key=lambda x: -x[1])[:8]:
    print(f"  {s}: {n}")

print("\nCOLOR MODES")
for m, n in modes.items():
    print(f"  {m}: {n}")

print("\nCORRUPT FILES:", len(corrupt))
for p, e in corrupt[:10]:
    print(" ", p, e)
    