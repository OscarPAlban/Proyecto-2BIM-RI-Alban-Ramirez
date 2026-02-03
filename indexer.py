import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import os

# 1. Cargar Modelo CLIP (Multimodal)
print("Cargando modelo CLIP...")
model = SentenceTransformer('clip-ViT-B-32')

# 2. Cargar datos
df = pd.read_pickle("metadata_clean.pkl")
img_dir = "images"

# 3. Generar Embeddings de las IMÁGENES
# El truco: Indexamos las imágenes, pero CLIP permite buscarlas con TEXTO o IMAGEN.
print("Generando embeddings...")
image_paths = [os.path.join(img_dir, f"{i}.jpg") for i in df.index]
images = [Image.open(p) for p in image_paths]

# Embeddings normalizados para usar Producto Interno como Similitud Coseno
img_embeddings = model.encode(images, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

# 4. Crear Índice FAISS
d = img_embeddings.shape[1] # Dimensión (512 para ViT-B-32)
index = faiss.IndexFlatIP(d) # IndexFlatIP es exacto y rápido para < 10k items
index.add(img_embeddings)

# 5. Guardar
faiss.write_index(index, "amazon.index")
print("Índice guardado exitosamente.")