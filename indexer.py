import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import os

# 1. Cargar Modelo CLIP 
print("Cargando modelo CLIP...")
model = SentenceTransformer('clip-ViT-B-32')

# 2. Cargar datos
df = pd.read_pickle("metadata_clean.pkl")
img_dir = "images"

# 3. Generar Embeddings de las IMÁGENES
print("Generando embeddings...")
image_paths = [os.path.join(img_dir, f"{i}.jpg") for i in df.index]
images = [Image.open(p) for p in image_paths]

img_embeddings = model.encode(images, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

# 4. Crear Índice FAISS
d = img_embeddings.shape[1] 
index = faiss.IndexFlatIP(d) 
index.add(img_embeddings)

# 5. Guardar
faiss.write_index(index, "amazon.index")
print("Índice guardado exitosamente.")