from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import openai
import pickle
import numpy as np
import os
import requests

# === CONFIGURAR DESDE VARIABLES DE ENTORNO (Azure) ===
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BLOB_URL = os.environ.get("EMBEDDINGS_BLOB_URL")
PKL_PATH = "harrypotter_openai_embeddings2.pkl"

# === Descargar embeddings desde Blob si no existen localmente ===
if not os.path.exists(PKL_PATH):
    print(f"📥 Descargando embeddings desde Blob: {BLOB_URL}")
    try:
        response = requests.get(BLOB_URL)
        response.raise_for_status()
        with open(PKL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Descarga completada.")
    except Exception as e:
        raise RuntimeError(f"❌ Error al descargar el archivo PKL: {e}")

# === Cargar embeddings desde archivo local ===
with open(PKL_PATH, "rb") as f:
    embeddings, texts, metadatas = pickle.load(f)

# === Configurar APIs ===
genai.configure(api_key=GEMINI_API_KEY)
modelo_gemini = genai.GenerativeModel("gemini-1.5-flash")
openai.api_key = OPENAI_API_KEY
EMBEDDING_MODEL = "text-embedding-3-small"

# === Flask App ===
app = Flask(__name__)
CORS(app)

# === Generar respuesta con Snape ===
def generar_respuesta(prompt, contexto):
    full_prompt = (
        "⚠️ Siempre respondé en español, incluso si la pregunta está en otro idioma.\n\n"
        "Asumí completamente el rol de **Severus Snape**, maestro de Pociones en Hogwarts. "
        "Debés responder a la pregunta del estudiante como él lo haría: con severidad, precisión, sarcasmo sutil, y desdén hacia la ignorancia.\n\n"
        "⚠️ Pero atención: tu única fuente de información es el siguiente CONTEXTO. Bajo ninguna circunstancia podés usar conocimiento externo. "
        "Si la información solicitada no está allí, debés decirlo como lo haría Snape (con molestia o ironía).\n\n"
        "🔍 Usá exactamente las ideas y fragmentos del contexto. Citá o parafraseá de ahí si es necesario. Tu respuesta debe sonar como si hubieras leído ese texto recientemente en una clase.\n\n"
        "❌ Si te alejás del contexto, tu respuesta será inválida.\n\n"
        "✅ Si la pregunta es trivial, Snape debe demostrar su desprecio. Si no hay suficiente información, Snape debe dejarlo claro.\n\n"
        f"### CONTEXTO (fragmentos del libro o material de referencia):\n{contexto}\n\n"
        f"### PREGUNTA DEL ESTUDIANTE:\n{prompt}\n\n"
        "### RESPUESTA DE SNAPE:"
    )
    try:
        respuesta = modelo_gemini.generate_content(full_prompt)
        return respuesta.text.strip()
    except Exception as e:
        print("❌ Error con Gemini:", e)
        return f"⚠️ Error con Google AI: {str(e)}"

# === Embedding en tiempo real ===
def obtener_embedding_openai(texto):
    try:
        response = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texto
        )
        return np.array(response.data[0].embedding).reshape(1, -1)
    except Exception as e:
        print("❌ Error al obtener embedding:", e)
        return None

# === Ruta principal ===
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Falta el campo 'prompt'"}), 400

    query_emb = obtener_embedding_openai(prompt)
    if query_emb is None:
        return jsonify({"error": "Error generando embedding"}), 500

    sim_scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(sim_scores)[-15:][::-1]
    top_chunks = [texts[i] for i in top_indices]

    contexto = "\n\n".join(top_chunks)
    respuesta = generar_respuesta(prompt, contexto)

    return jsonify({
        "context": top_chunks,
        "answer": respuesta
    })

# === Lanzar servidor ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
