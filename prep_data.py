import kagglehub
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
import glob

OUTPUT_FOLDER = "images"
METADATA_FILE = "metadata_clean.pkl"
MAX_REAL_PRODUCTS = 2000  

datos_examen = [
    {"title": "Silla de Oficina Ergonómica", "category": "Muebles", "q": "office-chair"},
    {"title": "Zapatillas Running Deportivas", "category": "Calzado", "q": "running-shoes"},
    {"title": "Botella de Agua Metalica", "category": "Deportes", "q": "water-bottle"},
    {"title": "Gafas de Sol Estilo RayBan", "category": "Accesorios", "q": "sunglasses"},
    {"title": "Reloj de Pulsera Clásico", "category": "Electrónica", "q": "wrist-watch"},
    {"title": "Mochila Escolar / Laptop", "category": "Accesorios", "q": "backpack"},
    {"title": "Taza de Café Cerámica", "category": "Hogar", "q": "coffee-mug"},
    {"title": "Teclado Gamer Mecánico", "category": "Electrónica", "q": "mechanical-keyboard"},
    {"title": "Guitarra Acústica Madera", "category": "Música", "q": "acoustic-guitar"},
    {"title": "Lámpara de Escritorio LED", "category": "Hogar", "q": "desk-lamp"}
]

def preparar_dataset():
    print(" Extrayendo productos ...")

    path = kagglehub.dataset_download("datafiniti/consumer-reviews-of-amazon-products")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    csv_path = max(csv_files, key=os.path.getsize)

    df = pd.read_csv(csv_path, encoding='latin1')

    df['title'] = df['name']
    df['category'] = df.get('primaryCategories', 'General')
    df['image_url'] = df['imageURLs']

    df = df.dropna(subset=['image_url'])
    print("Filas con imageURLs:", len(df))

    df_products = (
        df[['title', 'category', 'image_url']]
        .groupby('image_url', as_index=False)
        .first()
    )

    df_real = df_products.head(MAX_REAL_PRODUCTS).copy()

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(" Descargando imágenes...")

    valid_rows = []
    img_index = 0

    for _, row in df_real.iterrows():
        urls = str(row['image_url']).split(',')

        for url in urls:
            url = url.strip()
            if not url.startswith('http'):
                continue

            try:
                response = requests.get(url, timeout=4)
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img.thumbnail((300, 300))

                img.save(os.path.join(OUTPUT_FOLDER, f"{img_index}.jpg"))

                valid_rows.append({
                    'title': row['title'],
                    'category': row['category'],
                    'image_url': url
                })

                img_index += 1

            except:
                continue

    # Inyección para el examen
    for item in datos_examen:
        for i in range(1, 4):
            try:
                url = f"https://source.unsplash.com/300x300/?{item['q']}&sig={i+100}"
                response = requests.get(url, timeout=4)
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img.thumbnail((300, 300))

                img.save(os.path.join(OUTPUT_FOLDER, f"{img_index}.jpg"))

                valid_rows.append({
                    'title': f"{item['title']} - Var.{i}",
                    'category': item['category'],
                    'image_url': url
                })

                img_index += 1
            except:
                continue

    df_final = pd.DataFrame(valid_rows)
    df_final.to_pickle(METADATA_FILE)

    print(f" Imágenes guardadas: {img_index}")

if __name__ == "__main__":
    preparar_dataset()
