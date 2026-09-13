import streamlit as st
import joblib
import pandas as pd


st.set_page_config(
    page_title="Análisis de Sentimientos IMDb",
    page_icon="🎬",
    layout="wide"
)


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

        .palabra-clave {
            display: inline-block;
            padding: 6px 10px;
            margin: 4px;
            border-radius: 10px;
            background: #f2f2f2;
            font-size: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_sentimientos_imdb.pkl")


modelo = cargar_modelo()


def obtener_componentes(modelo_pipeline):
    vectorizador = modelo_pipeline.named_steps["vectorizador"]
    clasificador = modelo_pipeline.named_steps["clasificador"]
    return vectorizador, clasificador


def analizar_palabras(texto, modelo_pipeline, cantidad=10):
    vectorizador, clasificador = obtener_componentes(modelo_pipeline)

    matriz_tfidf = vectorizador.transform([texto])
    nombres = vectorizador.get_feature_names_out()
    coeficientes = clasificador.coef_[0]

    indices_presentes = matriz_tfidf.nonzero()[1]

    filas = []

    for indice in indices_presentes:
        valor_tfidf = float(matriz_tfidf[0, indice])
        peso_modelo = float(coeficientes[indice])
        contribucion = valor_tfidf * peso_modelo

        filas.append(
            {
                "Palabra": nombres[indice],
                "TF-IDF": valor_tfidf,
                "Peso del modelo": peso_modelo,
                "Contribución": contribucion,
            }
        )

    if not filas:
        return pd.DataFrame(
            columns=["Palabra", "TF-IDF", "Peso del modelo", "Contribución"]
        )

    df = pd.DataFrame(filas)
    df["Influencia absoluta"] = df["Contribución"].abs()

    df = (
        df.sort_values("Influencia absoluta", ascending=False)
        .head(cantidad)
        .drop(columns="Influencia absoluta")
    )

    return df


with st.sidebar:
    st.header("ℹ️ Sobre la aplicación")

    st.write(
        "Esta aplicación clasifica reseñas de películas en inglés "
        "como positivas o negativas."
    )

    st.markdown("**Modelo utilizado**")
    st.write("TF-IDF + LinearSVC")

    st.markdown("**Fuente de entrenamiento**")
    st.write("IMDb Dataset")

    st.markdown("**Salida del modelo**")
    st.write("Positive / Negative")

    st.divider()

    st.caption(
        "El margen de decisión del SVM no representa una probabilidad. "
        "Indica qué tan lejos se encuentra el texto de la frontera de clasificación."
    )


st.title("🎬 Análisis de Sentimientos de Reseñas")

st.write(
    "Ingresa una reseña de una película en inglés. "
    "La aplicación determinará si el sentimiento es positivo o negativo "
    "y mostrará qué palabras tuvieron mayor influencia en la clasificación."
)


texto = st.text_area(
    "Escribe la reseña:",
    height=180,
    placeholder=(
        "Ejemplo: This movie was amazing. "
        "The story was excellent and the actors were great."
    )
)

analizar = st.button(
    "🔎 Analizar sentimiento",
    type="primary",
    use_container_width=True
)


if analizar:

    if texto.strip() == "":
        st.warning("Por favor, escribe una reseña antes de realizar el análisis.")

    else:
        prediccion = modelo.predict([texto])[0]

        vectorizador, clasificador = obtener_componentes(modelo)
        margen = float(modelo.decision_function([texto])[0])

        clase_negativa = str(clasificador.classes_[0])
        clase_positiva = str(clasificador.classes_[1])

        st.divider()

        if str(prediccion).lower() == "positive":
            st.markdown(
                """
                <div class="resultado-positivo">
                    <div class="titulo-resultado">😊 SENTIMIENTO POSITIVO</div>
                    <div class="subtitulo">
                        El modelo interpreta la reseña como predominantemente positiva.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="resultado-negativo">
                    <div class="titulo-resultado">🙁 SENTIMIENTO NEGATIVO</div>
                    <div class="subtitulo">
                        El modelo interpreta la reseña como predominantemente negativa.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Clasificación",
                str(prediccion).upper()
            )

        with col2:
            st.metric(
                "Margen de decisión",
                f"{margen:.3f}"
            )

        with col3:
            st.metric(
                "Palabras ingresadas",
                len(texto.split())
            )

        st.caption(
            f"En este modelo, un margen mayor que 0 favorece «{clase_positiva}» "
            f"y un margen menor que 0 favorece «{clase_negativa}»."
        )

        st.subheader("🔑 Palabras con mayor influencia")

        importancia = analizar_palabras(
            texto,
            modelo,
            cantidad=10
        )

        if importancia.empty:
            st.info(
                "No se encontraron términos reconocidos por el vocabulario "
                "TF-IDF del modelo."
            )

        else:
            etiquetas = " ".join(
                f'<span class="palabra-clave">{palabra}</span>'
                for palabra in importancia["Palabra"].tolist()
            )

            st.markdown(
                etiquetas,
                unsafe_allow_html=True
            )

            st.write("")

            st.subheader("📊 Influencia de las palabras en la decisión")

            grafico = importancia[
                ["Palabra", "Contribución"]
            ].copy()

            grafico = grafico.sort_values(
                "Contribución",
                ascending=True
            )

            st.bar_chart(
                grafico.set_index("Palabra"),
                horizontal=True
            )

            st.caption(
                f"Valores positivos empujan la predicción hacia «{clase_positiva}». "
                f"Valores negativos la empujan hacia «{clase_negativa}»."
            )

            with st.expander("Ver detalle técnico"):
                tabla = importancia.copy()

                tabla["TF-IDF"] = tabla["TF-IDF"].round(4)
                tabla["Peso del modelo"] = tabla["Peso del modelo"].round(4)
                tabla["Contribución"] = tabla["Contribución"].round(4)

                st.dataframe(
                    tabla,
                    use_container_width=True,
                    hide_index=True
                )

                st.write(
                    "**Interpretación:** la contribución se calcula multiplicando "
                    "el valor TF-IDF que tiene una palabra en la reseña por el peso "
                    "que LinearSVC aprendió para esa palabra durante el entrenamiento."
                )


st.divider()

st.caption(
    "Proyecto de clasificación de sentimientos utilizando "
    "Procesamiento de Lenguaje Natural y Machine Learning."
)
