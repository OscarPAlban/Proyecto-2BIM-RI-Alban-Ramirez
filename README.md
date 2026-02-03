# **Sistema de Recuperación de Información Multimodal**

Este proyecto es un chat de recuperación de la Información Multimodal, que permite buscar productos en un inventario utilizando texto o imágenes. Utiliza **CLIP** para el entendimiento visual, **FAISS** para la búsqueda vectorial eficiente y **Gemini 2.0 Flash** para la generación de respuestas contextuales con memoria.

---

## **Requisitos del Sistema**

* **Python 3.9** o superior.
* Una **Google API Key** (Gemini).
* Conexión a internet para la descarga inicial de modelos y datos.

---

## **Instalación de Dependencias**

Ejecuta el siguiente comando en tu terminal para instalar todas las librerías necesarias:

pip install streamlit faiss-cpu pandas google-genai python-dotenv pillow sentence-transformers requests kagglehub

---
## **Configuración**

* Crea un archivo llamado .env en la carpeta raíz del proyect.
* Agrega tu llave de API de la siguiente manera:
GOOGLE_API_KEY=TU_LLAVE_AQUI

## **Orden de ejecución**

**En una terminal dentro del directorio escribe:**

* python prep_data.py
* python indexer.py
* streamlit run app.py
* **Opcional: streamlit run reranking_demo.py** (Para ver el reranking)



