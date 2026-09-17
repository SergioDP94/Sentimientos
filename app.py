import re
import html

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Análisis de Sentimientos con Embeddings",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .resultado-positivo {
            padding: 20px;
            border-radius: 14px;
            background-color: #e8f5e9;
            border: 1px solid #a5d6a7;
            text-align: center;
            margin-bottom: 15px;
        }

        .resultado-negativo {
            padding: 20px;
            border-radius: 14px;
            background-color: #ffebee;
            border: 1px solid #ef9a9a;
            text-align: center;
            margin-bottom: 15px;
        }

        .titulo-resultado {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitulo {
            color: #666666;
            font-size: 14px;
        }

        .caja-explicacion {
            padding: 14px 16px;
            border: 1px solid #dddddd;
            border-radius: 12px;
            margin-top: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CARGA
# ============================================================

@st.cache_resource
def cargar_archivo_modelo():
    return joblib.load(
        "modelo_sentimientos_imdb.pkl"
    )


modelo = cargar_archivo_modelo()

clasificador = modelo["clasificador"]
nombre_embedding = modelo["embedding_model_name"]
dimension_embedding = modelo["dimension_embedding"]
pca = modelo["pca"]

embeddings_referencia = modelo[
    "embeddings_referencia"
]

proyeccion_referencia = modelo[
    "proyeccion_referencia"
]

textos_referencia = modelo[
    "textos_referencia"
]

sentimientos_referencia = modelo[
    "sentimientos_referencia"
]

centroides = modelo[
    "centroides"
]


@st.cache_resource
def cargar_sentence_transformer(nombre):
    return SentenceTransformer(
        nombre
    )


modelo_embedding = cargar_sentence_transformer(
    nombre_embedding
)


# ============================================================
# FUNCIONES
# ============================================================

def generar_embedding(texto):
    return modelo_embedding.encode(
        [texto],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")


def limpiar_texto_visual(texto, longitud=160):
    texto = re.sub(
        r"<[^>]+>",
        " ",
        str(texto)
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    if len(texto) > longitud:
        texto = texto[:longitud] + "..."

    return texto


def obtener_vecinos(embedding, cantidad=8):

    vector = embedding[0]

    # Los embeddings están normalizados.
    # El producto punto equivale a similitud coseno.
    similitudes = embeddings_referencia @ vector

    indices = np.argsort(
        similitudes
    )[::-1][:cantidad]

    filas = []

    for posicion, indice in enumerate(
        indices,
        start=1
    ):
        filas.append(
            {
                "N.º": posicion,
                "Reseña": limpiar_texto_visual(
                    textos_referencia[indice]
                ),
                "Sentimiento": str(
                    sentimientos_referencia[indice]
                ).capitalize(),
                "Similitud": float(
                    similitudes[indice]
                )
            }
        )

    return pd.DataFrame(
        filas
    ), indices, similitudes[indices]


def similitud_centroides(embedding):

    vector = embedding[0]

    filas = []

    for clase, centroide in centroides.items():

        similitud = float(
            np.dot(
                vector,
                centroide
            )
        )

        filas.append(
            {
                "Sentimiento": str(
                    clase
                ).capitalize(),
                "Similitud": similitud
            }
        )

    return pd.DataFrame(
        filas
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "ℹ️ Sobre la aplicación"
    )

    st.write(
        "La reseña se transforma en un embedding semántico "
        "y luego se clasifica como positiva o negativa."
    )

    st.markdown(
        "**Embedding**"
    )
    st.write(
        "all-MiniLM-L6-v2"
    )

    st.markdown(
        "**Dimensión original**"
    )
    st.write(
        f"{dimension_embedding} dimensiones"
    )

    st.markdown(
        "**Clasificador**"
    )
    st.write(
        "LinearSVC"
    )

    st.divider()

    st.caption(
        "PCA se utiliza únicamente para representar los embeddings "
        "en dos dimensiones. El clasificador continúa trabajando "
        "con el vector completo."
    )


# ============================================================
# ENTRADA
# ============================================================

st.title(
    "🎬 Análisis de Sentimientos con Embeddings"
)

st.write(
    "Ingresa una reseña en inglés. Además de predecir su sentimiento, "
    "la aplicación mostrará cómo se representa semánticamente y qué "
    "reseñas del espacio de embeddings son más parecidas."
)


texto = st.text_area(
    "Escribe la reseña:",
    height=170,
    placeholder=(
        "This movie was amazing. "
        "The story was excellent and the actors were great."
    )
)


analizar = st.button(
    "🔎 Analizar sentimiento",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANÁLISIS
# ============================================================

if analizar:

    if texto.strip() == "":

        st.warning(
            "Por favor, escribe una reseña."
        )

    else:

        embedding = generar_embedding(
            texto
        )

        prediccion = clasificador.predict(
            embedding
        )[0]

        margen = float(
            clasificador.decision_function(
                embedding
            )[0]
        )

        nueva_proyeccion = pca.transform(
            embedding
        )[0]


        # ====================================================
        # RESULTADO
        # ====================================================

        st.divider()

        if str(prediccion).lower() == "positive":

            st.markdown(
                """
                <div class="resultado-positivo">
                    <div class="titulo-resultado">
                        😊 SENTIMIENTO POSITIVO
                    </div>
                    <div class="subtitulo">
                        La reseña fue clasificada como positiva.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="resultado-negativo">
                    <div class="titulo-resultado">
                        🙁 SENTIMIENTO NEGATIVO
                    </div>
                    <div class="subtitulo">
                        La reseña fue clasificada como negativa.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        col1, col2, col3 = st.columns(
            3
        )

        with col1:
            st.metric(
                "Clasificación",
                str(
                    prediccion
                ).upper()
            )

        with col2:
            st.metric(
                "Margen SVM",
                f"{margen:.3f}"
            )

        with col3:
            st.metric(
                "Dimensiones",
                dimension_embedding
            )


        # ====================================================
        # TABS
        # ====================================================

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "🗺️ Mapa semántico",
                "🔎 Reseñas similares",
                "🎯 Comparación por sentimiento",
                "🧬 Vector del embedding"
            ]
        )


        # ====================================================
        # TAB 1: MAPA PCA
        # ====================================================

        with tab1:

            st.subheader(
                "Mapa semántico en 2 dimensiones"
            )

            df_mapa = pd.DataFrame(
                {
                    "PCA 1": proyeccion_referencia[:, 0],
                    "PCA 2": proyeccion_referencia[:, 1],
                    "Sentimiento": [
                        str(x).capitalize()
                        for x in sentimientos_referencia
                    ],
                    "Reseña": [
                        limpiar_texto_visual(
                            x,
                            longitud=100
                        )
                        for x in textos_referencia
                    ]
                }
            )

            fig = px.scatter(
                df_mapa,
                x="PCA 1",
                y="PCA 2",
                color="Sentimiento",
                hover_data={
                    "Reseña": True,
                    "PCA 1": ":.3f",
                    "PCA 2": ":.3f"
                },
                opacity=0.55,
                title=(
                    "Distribución de reseñas en el espacio de embeddings"
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[nueva_proyeccion[0]],
                    y=[nueva_proyeccion[1]],
                    mode="markers",
                    name="Nueva reseña",
                    marker=dict(
                        size=18,
                        symbol="star",
                        color="black",
                        line=dict(
                            width=1,
                            color="white"
                        )
                    ),
                    hovertemplate=(
                        "<b>Nueva reseña</b><extra></extra>"
                    )
                )
            )

            fig.update_layout(
                height=620
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "PCA reduce los 384 valores del embedding a dos ejes "
                "para poder observar visualmente la posición relativa "
                "de las reseñas. La estrella representa el texto ingresado."
            )


        # ====================================================
        # TAB 2: VECINOS
        # ====================================================

        with tab2:

            st.subheader(
                "Reseñas semánticamente más cercanas"
            )

            df_vecinos, indices_vecinos, similitudes = obtener_vecinos(
                embedding,
                cantidad=8
            )

            grafico_vecinos = df_vecinos.copy()

            grafico_vecinos[
                "Etiqueta"
            ] = [
                f"Reseña {i}"
                for i in grafico_vecinos["N.º"]
            ]

            fig_vecinos = px.bar(
                grafico_vecinos.sort_values(
                    "Similitud"
                ),
                x="Similitud",
                y="Etiqueta",
                orientation="h",
                color="Sentimiento",
                range_x=[
                    max(
                        0,
                        float(
                            grafico_vecinos[
                                "Similitud"
                            ].min()
                        ) - 0.05
                    ),
                    1
                ],
                title="Similitud coseno con la nueva reseña"
            )

            fig_vecinos.update_layout(
                height=450
            )

            st.plotly_chart(
                fig_vecinos,
                use_container_width=True
            )

            tabla_vecinos = df_vecinos.copy()

            tabla_vecinos[
                "Similitud"
            ] = tabla_vecinos[
                "Similitud"
            ].round(4)

            st.dataframe(
                tabla_vecinos,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Una similitud más cercana a 1 indica que las reseñas "
                "están más próximas dentro del espacio semántico."
            )


        # ====================================================
        # TAB 3: CENTROIDES
        # ====================================================

        with tab3:

            st.subheader(
                "Cercanía semántica a cada sentimiento"
            )

            df_centroides = similitud_centroides(
                embedding
            )

            fig_centroides = px.bar(
                df_centroides,
                x="Sentimiento",
                y="Similitud",
                color="Sentimiento",
                text_auto=".3f",
                title=(
                    "Similitud de la reseña con el centro "
                    "semántico de cada clase"
                )
            )

            fig_centroides.update_layout(
                showlegend=False,
                height=430
            )

            st.plotly_chart(
                fig_centroides,
                use_container_width=True
            )

            st.caption(
                "Cada barra compara el embedding de la nueva reseña "
                "con el embedding promedio de las reseñas positivas "
                "y negativas del conjunto de entrenamiento."
            )


        # ====================================================
        # TAB 4: VECTOR
        # ====================================================

        with tab4:

            st.subheader(
                "Perfil del embedding"
            )

            vector = embedding[0]

            cantidad = min(
                80,
                len(vector)
            )

            df_vector = pd.DataFrame(
                {
                    "Dimensión": np.arange(
                        1,
                        cantidad + 1
                    ),
                    "Valor": vector[
                        :cantidad
                    ]
                }
            )

            fig_vector = px.line(
                df_vector,
                x="Dimensión",
                y="Valor",
                markers=False,
                title=(
                    f"Primeras {cantidad} dimensiones "
                    "del vector generado"
                )
            )

            fig_vector.update_layout(
                height=430
            )

            st.plotly_chart(
                fig_vector,
                use_container_width=True
            )

            st.caption(
                "Cada dimensión es una característica latente aprendida "
                "por el modelo. No debe interpretarse de forma aislada; "
                "el significado aparece en el patrón completo del vector."
            )


        # ====================================================
        # FLUJO
        # ====================================================

        st.divider()

        st.subheader(
            "🔄 Flujo del modelo"
        )

        st.code(
            """
Reseña
   ↓
SentenceTransformer
   ↓
Embedding de 384 dimensiones
   ├──→ PCA → mapa semántico
   ├──→ similitud coseno → vecinos
   └──→ LinearSVC → Positive / Negative
            """,
            language=None
        )


st.divider()

st.caption(
    "Proyecto de clasificación de sentimientos con embeddings, "
    "visualización semántica y Machine Learning."
)
