import streamlit as st
import joblib


# Configuración de la página
st.set_page_config(
    page_title="Análisis de Sentimientos",
    page_icon="🎬",
    layout="centered"
)


# Cargar modelo
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_sentimientos_imdb.pkl")


modelo = cargar_modelo()


# Título
st.title("🎬 Análisis de Sentimientos")

st.write(
    "Escribe una reseña de una película en inglés "
    "y el modelo determinará si es positiva o negativa."
)


# Caja de texto
texto = st.text_area(
    "Escribe la reseña:",
    height=180,
    placeholder="Ejemplo: This movie was amazing. The story was excellent..."
)


# Botón
if st.button("Analizar sentimiento"):

    if texto.strip() == "":
        st.warning("Por favor, escribe una reseña.")

    else:

        resultado = modelo.predict([texto])[0]

        if resultado == "positive":
            st.success("😊 Sentimiento: POSITIVO")

        else:
            st.error("🙁 Sentimiento: NEGATIVO")