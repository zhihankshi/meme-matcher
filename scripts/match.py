import json
from pathlib import Path
import numpy as np
from transformers import CLIPProcessor, CLIPModel

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = ROOT / "embeddings"

def load_embeddings():
    data = np.load(EMBEDDINGS_DIR / "embeddings.npz", allow_pickle=True)
    if "names" in data and "vectors" in data:
        return dict(zip(data["names"].tolist(), data["vectors"]))
    return dict(data["embedded_images"].tolist())

def match(query, top_k=5):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # embed the query
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    query_embedding = model.get_text_features(**inputs).detach().numpy().squeeze()
    
    # load image embeddings
    embeddings = load_embeddings()
    
    # calculate similarities
    results = []
    for img_name, img_embedding in embeddings.items():
        similarity = np.dot(query_embedding, img_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(img_embedding)
        )
        results.append((img_name, float(similarity)))
    
    # sort and return top matches
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    while True:
        query = input("\nWhat do you want to say? (or 'quit'): ").strip()
        if query.lower() == "quit":
            break
        
        matches = match(query)
        print("\nTop matches:")
        for img, score in matches:
            print(f"  {img}: {score:.3f}")