# IMDb NLP Processor

Proyecto académico para la **Unidad 2 - Clase 1: Text Mining y Embeddings Contextuales** del módulo **Científico de Datos e Inteligencia Artificial Aplicada**.

## Descripción del proyecto

Este proyecto implementa un flujo completo y modular de preparación de texto sobre el **IMDb Movie Reviews Dataset (Kaggle)**. El foco está en comprender el pipeline previo al modelado avanzado:

1. Exploración del dataset (EDA)
2. Limpieza de texto
3. Tokenización
4. Conversión a secuencias numéricas
5. Padding
6. Introducción a Word Embeddings con TensorFlow/Keras

No se utilizan Transformers, BERT, GPT, Hugging Face ni modelos preentrenados, ya que esos temas corresponden a clases posteriores.

## Objetivos

- Trabajar con un dataset real de reseñas de películas.
- Aplicar técnicas fundamentales de NLP para texto crudo.
- Documentar decisiones técnicas del preprocesamiento.
- Construir una base reutilizable para evolucionar hacia arquitecturas modernas en la Unidad 2.

## Dataset

- **Nombre:** IMDb Movie Reviews Dataset
- **Fuente:** Kaggle
- **Columnas esperadas:**
  - `review` (texto de la reseña)
  - `sentiment` (`positive` o `negative`)

### Instrucciones de carga del dataset

1. Descarga el dataset desde Kaggle.
2. Exporta o renombra el CSV a `imdb_reviews.csv`.
3. Colócalo en [dataset/imdb_reviews.csv](dataset/imdb_reviews.csv).

Ejemplo con Kaggle CLI (si ya configuraste `kaggle.json`):

```bash
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews -p dataset --unzip
```

Si el archivo descargado se llama `IMDB Dataset.csv`, renómbralo a `imdb_reviews.csv`.

> Nota: para acelerar la práctica, el notebook permite trabajar con una muestra (por ejemplo 2,000 a 5,000 registros).

## Estructura del proyecto

```text
IMDb_NLP_Processor/
├── README.md
├── requirements.txt
├── dataset/
│   └── imdb_reviews.csv
├── notebooks/
│   └── actividad.ipynb
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── tokenizer_utils.py
│   ├── embedding_utils.py
│   └── utils.py
└── images/
```

## Instalación

Recomendado: Python 3.12 para compatibilidad estable con TensorFlow.

```bash
pip install -r requirements.txt
```

## Ejecución

1. Asegura que el CSV esté en [dataset/imdb_reviews.csv](dataset/imdb_reviews.csv).
2. Abre Jupyter Notebook.
3. Ejecuta [notebooks/actividad.ipynb](notebooks/actividad.ipynb) de principio a fin.

## Resultados esperados en el notebook

- Resumen estructural del dataset.
- Distribución de sentimientos y longitudes de reseñas.
- Ejemplos antes y después de limpieza.
- Tamaño de vocabulario y top 20 palabras más frecuentes.
- Secuencias numéricas y variantes de padding (pre/post).
- `model.summary()` de una red mínima con Embedding.
- Reflexión técnica argumentada.

## Capturas de pantalla sugeridas

Guardar evidencias en [images/](images/), por ejemplo:

- `eda_distribucion_sentimientos.png`
- `eda_hist_longitud.png`
- `limpieza_antes_despues.png`
- `tokenizacion_ejemplos.png`
- `padding_comparacion.png`
- `embedding_summary.png`

## Conclusiones

Incluye en tu entrega final:

- Qué decisiones de limpieza fueron más relevantes.
- Qué impacto observaste al cambiar parámetros de tokenización/padding.
- Qué aprendiste sobre el salto de texto a vectores.
- Cómo este repositorio se puede extender hacia Transformers y fine-tuning en clases posteriores.

## Entregables

1. Repositorio en GitHub.
2. Código fuente organizado.
3. Notebook completamente ejecutado.
4. README profesional.
5. Evidencias visuales en `images/`.

## Rúbrica de evaluación (100 puntos)

| Criterio | Puntaje |
|---|---:|
| Organización del proyecto | 10 |
| Exploración del dataset (EDA) | 15 |
| Limpieza del texto | 20 |
| Tokenización | 15 |
| Padding | 10 |
| Implementación de Embedding Layer | 10 |
| Calidad del código | 10 |
| README y documentación | 5 |
| Reflexión técnica | 5 |

## Punto extra (+10)

Entrenar la red 3 a 5 épocas y reportar:

- Accuracy
- Loss
- Matriz de confusión
- Predicciones sobre nuevas reseñas
