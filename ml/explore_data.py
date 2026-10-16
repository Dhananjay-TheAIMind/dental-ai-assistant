import os, random
from PIL import Image

ROOT = os.path.join("data", "raw", "Dental_Radiography")
SPLITS = ["train", "valid", "test"]
random.seed(0)

def base_name(fname):
    return fname.split(".rf.")[0]

files = {}
for s in SPLITS:
    for c in sorted(os.listdir(os.path.join(ROOT, s))):
        d = os.path.join(ROOT, s, c)
        if os.path.isdir(d):
            files[(s, c)] = [f for f in os.listdir(d) if f.lower().endswith(".png")]

classes = sorted({c for (_, c) in files})

print("Image sizes (30 sampled per class, train):")
for c in classes:
    fl = files[("train", c)]
    sample = random.sample(fl, min(30, len(fl)))
    sizes = [Image.open(os.path.join(ROOT, "train", c, f)).size for f in sample]
    ws = [w for w, h in sizes]
    hs = [h for w, h in sizes]
    print(f"  {c:15s} width {min(ws)}-{max(ws)}  height {min(hs)}-{max(hs)}")

bases = {}
for s in SPLITS:
    names = set()
    for (sp, c), fl in files.items():
        if sp == s:
            for f in fl:
                names.add(base_name(f))
    bases[s] = names

print()
print("Unique source X-rays per split:", {s: len(b) for s, b in bases.items()})
print("Shared train & valid:", len(bases["train"] & bases["valid"]))
print("Shared train & test :", len(bases["train"] & bases["test"]))
print("Shared valid & test :", len(bases["valid"] & bases["test"]))

os.makedirs("docs", exist_ok=True)
T = 160
cols = 6
grid = Image.new("L", (T * cols, T * len(classes)), 0)
for i, c in enumerate(classes):
    fl = files[("train", c)]
    for j, f in enumerate(random.sample(fl, min(cols, len(fl)))):
        im = Image.open(os.path.join(ROOT, "train", c, f)).convert("L")
        im.thumbnail((T, T))
        grid.paste(im, (j * T, i * T))
grid.save(os.path.join("docs", "sample_grid.png"))
print()
print("Saved docs/sample_grid.png")
print("Rows from top to bottom:", ", ".join(classes))
