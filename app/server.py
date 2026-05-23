from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = ROOT / "embeddings"
IMAGES_DIR = ROOT / "images"

app = FastAPI()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def load_embeddings():
    data = np.load(EMBEDDINGS_DIR / "embeddings.npz", allow_pickle=True)
    if "names" in data and "vectors" in data:
        return dict(zip(data["names"].tolist(), data["vectors"]))
    return dict(data["embedded_images"].tolist())


embeddings = load_embeddings()


class Query(BaseModel):
    text: str


@app.post("/match")
def match(query: Query):
    inputs = processor(text=[query.text], return_tensors="pt", padding=True)
    query_embedding = model.get_text_features(**inputs).detach().numpy().squeeze()

    results = []
    for img_name, img_embedding in embeddings.items():
        similarity = np.dot(query_embedding, img_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(img_embedding)
        )
        results.append({"image": img_name, "score": float(similarity)})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:5]


app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "index.html")
