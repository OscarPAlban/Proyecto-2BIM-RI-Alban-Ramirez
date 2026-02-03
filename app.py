import streamlit as st
import faiss
import pandas as pd
import os
import hashlib
from google import genai
from dotenv import load_dotenv
from PIL import Image
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Buscador Visual Multimodal", layout="centered")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

@st.cache_resource
def load_resources():
    model = SentenceTransformer('clip-ViT-B-32')
    index = faiss.read_index("amazon.index")
    df = pd.read_pickle("metadata_clean.pkl")
    return model, index, df

model, index, df = load_resources()

if "chat" not in st.session_state:
    st.session_state.chat = []

if "last_image_hash" not in st.session_state:
    st.session_state.last_image_hash = None

def search(query_obj, modality='text'):
    if modality == 'text':
        query_emb = model.encode([query_obj], normalize_embeddings=True)
    else:
        query_emb = model.encode([query_obj], normalize_embeddings=True)

    D, I = index.search(query_emb, 10)
    candidates = df.iloc[I[0]].copy()

    imgs = []
    ids = []

    for idx in candidates.index:
        try:
            imgs.append(Image.open(f"images/{idx}.jpg"))
            ids.append(idx)
        except:
            pass

    scores = util.cos_sim(
        query_emb,
        model.encode(imgs, normalize_embeddings=True)
    )[0]

    candidates.loc[ids, 'score'] = scores.cpu().numpy()
    return candidates.sort_values('score', ascending=False)

def describe_user_image(img):
    prompt = "Describe en UNA sola línea qué objeto aparece en esta imagen en español."
    try:
        response = client.models.generate_content(
            model="gemini-3.0-flash",
            contents=[prompt, img]
        )
        return response.text.strip()
    except:
        return "Objeto no identificado visualmente."

def generate_rag(query, product, mode):
    prompt = f"""
Eres un sistema de inventario.

Usuario buscó: "{query}" (Modo: {mode})
Producto encontrado: "{product['title']}" (Categoría: {product['category']})

Responde en máximo 4 líneas:
• Qué objeto es
• Para qué sirve
• Categoría
• Si coincide con la búsqueda
"""
    try:
        r = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return r.text.strip()
    except Exception as e:
        return str(e)

def process_query(query, mode, img=None):
    user_image_description = None

    if img:
        user_image_description = describe_user_image(img)
        results = search(img, modality='image')
    else:
        results = search(query, modality='text')

    top = results.iloc[0]
    img_path = f"images/{top.name}.jpg"
    response = generate_rag(query, top, mode)

    st.session_state.chat.append({
        "role": "user",
        "query": query,
        "mode": mode,
        "image": img,
        "image_desc": user_image_description
    })

    # Sistema
    st.session_state.chat.append({
        "role": "assistant",
        "image_path": img_path,
        "title": top['title'],
        "response": response
    })

st.title("🔎 Sistema de Recuperación Multimodal")
st.markdown("Adjunta una imagen 🖼️ o escribe qué deseas buscar 👇")

prompt = st.chat_input(
    "Adjunta una imagen o escribe tu búsqueda...",
    accept_file=True,
    file_type=["jpg", "png"]
)


if prompt:
    text = prompt.text
    files = prompt.files

    # Imagen
    if files:
        image = Image.open(files[0])
        bytes_img = files[0].getvalue()
        img_hash = hashlib.md5(bytes_img).hexdigest()

        if img_hash != st.session_state.last_image_hash:
            process_query("Imagen enviada por el usuario", "Imagen", image)
            st.session_state.last_image_hash = img_hash
            st.rerun()

    # Texto
    elif text:
        process_query(text, "Texto")
        st.rerun()

# ---------------- CHAT DISPLAY ----------------
for msg in st.session_state.chat:
    if msg["role"] == "user":
        with st.chat_message("user"):
            if msg["mode"] == "Imagen":
                st.image(msg["image"], width=220)
                st.write(f"**Imagen del usuario:** {msg['image_desc']}")
            else:
                st.write(msg["query"])

    else:
        with st.chat_message("assistant"):
            st.image(msg["image_path"], width=260)
            st.caption(msg["title"])
            st.write(msg["response"])
