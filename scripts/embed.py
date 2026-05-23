# embed.py
import os
import json
from pathlib import Path
import numpy as np
from transformers import CLIPProcessor, CLIPModel

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMBEDDINGS_DIR = ROOT / "embeddings"

def load_existing_embeddings():
    path = EMBEDDINGS_DIR / "embeddings.npz"
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    if "names" in data and "vectors" in data:
        return dict(zip(data["names"].tolist(), data["vectors"]))
    # legacy format from earlier versions
    return dict(data["embedded_images"].tolist())


def save_embeddings(embeddings):
    names = np.array(list(embeddings.keys()))
    vectors = np.stack([np.asarray(v).squeeze() for v in embeddings.values()])
    np.savez(EMBEDDINGS_DIR / "embeddings.npz", names=names, vectors=vectors)

def embed_new_images():
    # load model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # load meme data
    with open(DATA_DIR / "memes.json", "r") as f:
        memes = json.load(f)
    
    # check existing
    existing = load_existing_embeddings()
    
    new_embeddings = {}
    
    for meme in memes:
        img_name = meme["image_path"]
        
        if img_name in existing:
            print(f"Skipping {img_name} - already embedded")
            continue
        
        print(f"Embedding {img_name}...")
        
        # embed the usage text, not the image itself
        inputs = processor(text=[meme["usage"]], return_tensors="pt", padding=True)
        text_features = model.get_text_features(**inputs)
        
        new_embeddings[img_name] = text_features.detach().numpy().squeeze()
    
    # merge and save
    all_embeddings = {**existing, **new_embeddings}
    save_embeddings(all_embeddings)
    
    print(f"Done! {len(new_embeddings)} new, {len(all_embeddings)} total")

if __name__ == "__main__":
    embed_new_images()