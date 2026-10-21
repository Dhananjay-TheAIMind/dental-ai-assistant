import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from PIL import Image
from backend.main import classify

dirs = [p for p in (ROOT / "data").rglob("Cavity")
        if p.is_dir() and p.parent.name.lower() == "test"]
print("Folders checked:", [str(d) for d in dirs])
files = [f for d in dirs for f in d.iterdir()
         if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
hits, misses = 0, {}
for f in files:
    _, top = classify(Image.open(f))
    if top == "Cavity":
        hits += 1
    else:
        misses[top] = misses.get(top, 0) + 1
print("Test cavity images:", len(files))
print("Correctly called Cavity: %d (%.0f%%)" % (hits, 100 * hits / max(1, len(files))))
print("Missed, predicted as:", misses)
