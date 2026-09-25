import pandas as pd
import numpy as np
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA


# ============================================================
# 1. CARGAR LOS DATOS
# ============================================================

datos = pd.read_csv("IMDB Dataset.csv")

# Muestra aleatoria para acelerar el procesamiento.
# random_state permite reproducir exactamente la misma muestra.

X = datos["review"].astype(str)
y = datos["sentiment"].astype(str)


# ============================================================
# 2. SEPARAR ENTRENAMIENTO Y PRUEBA
# ============================================================

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. MODELO DE EMBEDDINGS
# ============================================================

NOMBRE_MODELO_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"

modelo_embedding = SentenceTransformer(
    NOMBRE_MODELO_EMBEDDING
)


# ============================================================
# 4. GENERAR EMBEDDINGS
# ============================================================

print("Generando embeddings de entrenamiento...")

X_entrenamiento_emb = modelo_embedding.encode(
    X_entrenamiento.tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

print("Generando embeddings de prueba...")

X_prueba_emb = modelo_embedding.encode(
    X_prueba.tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")


# ============================================================
# 5. ENTRENAR CLASIFICADOR
# ============================================================

clasificador = LinearSVC()

clasificador.fit(
    X_entrenamiento_emb,
    y_entrenamiento
)


# ============================================================
# 6. EVALUAR
# ============================================================

predicciones = clasificador.predict(
    X_prueba_emb
)

accuracy = accuracy_score(
    y_prueba,
    predicciones
)

print("\nAccuracy:", round(accuracy, 4))
print("\nReporte de clasificación:")
print(
    classification_report(
        y_prueba,
        predicciones
    )
)


# ============================================================
# 7. PCA PARA VISUALIZACIÓN
# ============================================================
#
# PCA NO reemplaza los 384 valores usados por el clasificador.
# Solamente reduce el embedding a 2 dimensiones para visualizarlo.
# ============================================================

pca = PCA(
    n_components=2,
    random_state=42
)

pca.fit(
    X_entrenamiento_emb
)


# ============================================================
# 8. MUESTRA DE REFERENCIA PARA LA APP
# ============================================================
#
# No hace falta llevar todas las reseñas a Streamlit.
# Guardamos hasta 1000 observaciones representativas para:
# - mapa semántico
# - búsqueda de vecinos
# - comparaciones visuales
# ============================================================

rng = np.random.RandomState(42)

n_referencia = min(
    1000,
    len(X_entrenamiento_emb)
)

indices_referencia = rng.choice(
    len(X_entrenamiento_emb),
    size=n_referencia,
    replace=False
)

embeddings_referencia = X_entrenamiento_emb[
    indices_referencia
].astype("float32")

textos_referencia = X_entrenamiento.iloc[
    indices_referencia
].tolist()

sentimientos_referencia = y_entrenamiento.iloc[
    indices_referencia
].tolist()

proyeccion_referencia = pca.transform(
    embeddings_referencia
).astype("float32")


# ============================================================
# 9. CENTROIDES SEMÁNTICOS POR CLASE
# ============================================================

clases = np.unique(
    y_entrenamiento
)

centroides = {}

for clase in clases:

    mascara = (
        y_entrenamiento.to_numpy() == clase
    )

    centroide = X_entrenamiento_emb[
        mascara
    ].mean(axis=0)

    norma = np.linalg.norm(
        centroide
    )

    if norma > 0:
        centroide = centroide / norma

    centroides[str(clase)] = centroide.astype(
        "float32"
    )


# ============================================================
# 10. EXPORTAR
# ============================================================

modelo_exportado = {
    "embedding_model_name": NOMBRE_MODELO_EMBEDDING,
    "clasificador": clasificador,
    "dimension_embedding": int(
        X_entrenamiento_emb.shape[1]
    ),
    "pca": pca,
    "embeddings_referencia": embeddings_referencia,
    "proyeccion_referencia": proyeccion_referencia,
    "textos_referencia": textos_referencia,
    "sentimientos_referencia": sentimientos_referencia,
    "centroides": centroides,
    "accuracy": float(accuracy)
}

joblib.dump(
    modelo_exportado,
    "modelo_embeddings_imdb.pkl",
    compress=3
)

print("\nModelo guardado correctamente.")
print("Archivo: modelo_embeddings_imdb.pkl")
print("Dimensión:", X_entrenamiento_emb.shape[1])
print("Reseñas de referencia:", n_referencia)
