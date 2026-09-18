# Visual Analytics para reseñas cinematográficas IMDb

## Descripción

Proyecto de análisis exploratorio y Visual Analytics aplicado al
análisis de sentimientos en reseñas cinematográficas.

El objetivo es explorar patrones lingüísticos y semánticos dentro de un
corpus textual utilizando técnicas de Procesamiento de Lenguaje Natural.

## Dataset

IMDb Dataset of 50K Movie Reviews.

Variables principales:

-   review: reseña textual.
-   sentiment: clasificación positiva o negativa.

## Metodología

El proyecto sigue un pipeline de Ciencia de Datos:

1.  Carga y exploración de datos.
2.  Limpieza y preparación textual.
3.  Representación mediante TF-IDF.
4.  Generación de embeddings con Sentence Transformers.
5.  Reducción dimensional mediante PCA.
6.  Visualización y análisis semántico.

## Tecnologías

-   Python
-   Pandas
-   Scikit-learn
-   Sentence Transformers
-   Streamlit
-   Plotly

## Estructura del proyecto

    Proyecto/
    │
    ├── app.py
    ├── modelo_embeddings_imdb.pkl
    ├── requirements.txt
    │
    ├── notebooks/
    │   └── Analisis_Exploratorio_Visual_Analytics_IMDb.ipynb
    │
    └── data/
        └── IMDB Dataset.csv

## Aplicación Streamlit

La aplicación permite:

-   clasificar nuevas reseñas.
-   explorar el espacio semántico.
-   encontrar reseñas similares.
-   analizar patrones dentro del corpus.

## Autor

Sergio Díaz
