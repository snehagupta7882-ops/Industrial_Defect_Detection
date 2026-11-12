import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path("dataset")
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
NEU_CLASSES = {"crazing", "inclusion", "patches", "pitted_surface",
               "rolled-in_scale", "scratches"}

rows = []
for p in ROOT.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in EXTS:
        continue
    parts = p.relative_to(ROOT).parts   # material, condition, [defect], file
    material, condition = parts[0], parts[1]
    defect = parts[2] if condition == "defective" and len(parts) == 4 else "none"
    source = "neu" if defect in NEU_CLASSES else "mvtec"
    rows.append({"path": str(p), "material": material, "condition": condition,
                 "defect": defect, "source": source})

df = pd.DataFrame(rows)

# Har group (material + condition + source) ko 70/15/15 mein baanto
df["group"] = df["material"] + "|" + df["condition"] + "|" + df["source"]
train, temp = train_test_split(df, test_size=0.30, stratify=df["group"], random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["group"], random_state=42)

df["split"] = "train"
df.loc[val.index, "split"] = "val"
df.loc[test.index, "split"] = "test"

df.drop(columns="group").to_csv("dataset_split.csv", index=False)

print("Total:", len(df))
print(df.groupby(["split", "material", "condition"]).size().unstack(fill_value=0))