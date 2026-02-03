import streamlit as st
import faiss
import pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Demo Re-ranking", layout="wide")

st.title("🔬 Demostración del impacto del Re-ranking")

# ---------- CARGA ----------
@st.cache_resource
def load():
    model = SentenceTransformer('clip-ViT-B-32')
    index = faiss.read_index("amazon.index")
    df = pd.read_pickle("metadata_clean.pkl")
    return model, index, df

model, index, df = load()

# ---------- FUNCIÓN BASE FAISS ----------
def faiss_search(query, k=20):
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    candidates = df.iloc[I[0]].copy()
    return candidates, I[0]

# ---------- RE-RANKING REAL (IMAGEN ↔ IMAGEN) ----------
def rerank_visually(candidates, ids):
    # Tomamos la PRIMERA imagen como referencia visual
    ref_img = Image.open(f"images/{ids[0]}.jpg")
    ref_emb = model.encode(ref_img, normalize_embeddings=True)

    imgs = []
    valid_ids = []

    for idx in ids:
        try:
            img = Image.open(f"images/{idx}.jpg")
            imgs.append(img)
            valid_ids.append(idx)
        except:
            pass

    img_embs = model.encode(imgs, normalize_embeddings=True)
    scores = util.cos_sim(ref_emb, img_embs)[0]

    candidates = candidates.loc[valid_ids].copy()
    candidates["rerank_score"] = scores.cpu().numpy()

    return candidates.sort_values("rerank_score", ascending=False)

# ---------- UI ----------
query = st.text_input("Escribe una búsqueda para ver el efecto del re-ranking:", "tablet amazon fire")

if query:
    before, ids = faiss_search(query)
    after = rerank_visually(before, ids)

    st.divider()
    st.subheader("🔹 Antes del Re-ranking (FAISS: texto → imagen)")
    cols = st.columns(5)

    for i, (idx, row) in enumerate(before.head(5).iterrows()):
        with cols[i]:
            st.image(f"images/{idx}.jpg", use_container_width=True)
            st.caption(row['title'])

    st.divider()
    st.subheader("⭐ Después del Re-ranking (Imagen ↔ Imagen)")
    cols = st.columns(5)

    for i, (idx, row) in enumerate(after.head(5).iterrows()):
        with cols[i]:
            st.image(f"images/{idx}.jpg", use_container_width=True)
            st.caption(row['title'])
