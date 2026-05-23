import os
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
DATA_DIR = ROOT / "data"

SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".gif", ".webp")

def annotate():
    images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(SUPPORTED_FORMATS)]
    
    # load existing annotations to avoid redoing work
    json_path = DATA_DIR / "memes.json"
    existing = []
    if json_path.exists() and json_path.stat().st_size > 0:
        with open(json_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    annotated = {m["image_path"] for m in existing}
    
    new_annotations = []
    
    for img_name in images:
        if img_name in annotated:
            print(f"Skipping {img_name} - already annotated")
            continue
        
        # show the image
        img = Image.open(IMAGES_DIR / img_name)
        img.show()
        
        print(f"\n{img_name}")
        usage = input("Vibe/usage (comma separated): ").strip()
        
        if usage:
            new_annotations.append({
                "image_path": img_name,
                "usage": usage
            })
    
    # merge and save
    all_annotations = existing + new_annotations
    with open(json_path, "w") as f:
        json.dump(all_annotations, f, indent=2)
    
    print(f"\nDone! {len(new_annotations)} new, {len(all_annotations)} total")

if __name__ == "__main__":
    annotate()