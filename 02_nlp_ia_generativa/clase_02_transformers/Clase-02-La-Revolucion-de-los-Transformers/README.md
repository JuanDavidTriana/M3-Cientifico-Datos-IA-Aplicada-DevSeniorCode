# Clase 02: La Revolución de los Transformers

Repositorio de apoyo para una clase de aproximadamente 4-6 horas del módulo **Científico de Datos e Inteligencia Artificial Aplicada**.

## Estructura del proyecto

```text
Clase-02-La-Revolucion-de-los-Transformers/
│
├── README.md
├── requirements.txt
├── datasets/
│   ├── frases_atencion.csv
│   ├── reviews_sentiment.csv
│   ├── documentos_qa.csv
│   └── casos_empresariales.csv
└── notebooks/
    ├── 01_Historia_Transformers.ipynb
    ├── 02_Embeddings_Tokenizacion_SelfAttention.ipynb
    ├── 03_Visualizacion_Atencion.ipynb
    ├── 04_BERT_Aplicaciones.ipynb
    ├── 05_GPT_Generacion_Texto.ipynb
    ├── 06_Comparacion_Modelos.ipynb
    ├── 07_Aplicaciones_Empresariales.ipynb
    └── 08_Actividad_Final.ipynb
```

## Instalación de dependencias

1. Crear y activar un entorno virtual (opcional, recomendado).
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

> **Nota:** La primera ejecución descargará modelos preentrenados de Hugging Face (DistilBERT, GPT-2, etc.). Se recomienda conexión a internet estable.

## Cómo ejecutar los notebooks

Desde la raíz del proyecto:

```bash
jupyter notebook
```

Abrir la carpeta `notebooks/` y ejecutar cada notebook en orden.

## Orden recomendado de estudio

1. `01_Historia_Transformers.ipynb`
2. `02_Embeddings_Tokenizacion_SelfAttention.ipynb`
3. `03_Visualizacion_Atencion.ipynb`
4. `04_BERT_Aplicaciones.ipynb`
5. `05_GPT_Generacion_Texto.ipynb`
6. `06_Comparacion_Modelos.ipynb`
7. `07_Aplicaciones_Empresariales.ipynb`
8. `08_Actividad_Final.ipynb`

## Objetivos de aprendizaje por notebook

- **01 Historia de los Transformers**: comprender la evolución desde RNN/LSTM hasta la arquitectura Transformer.
- **02 Embeddings, Tokenización, Self Attention**: implementar Scaled Dot-Product Attention desde cero.
- **03 Visualización de Atención**: interpretar matrices de atención con heatmaps.
- **04 BERT**: clasificación de sentimientos, QA y NER con modelos preentrenados.
- **05 GPT**: generación de texto, resúmenes y parámetros de muestreo (temperatura, top-p, top-k).
- **06 Comparación de Modelos**: contrastar BERT, GPT, T5, Llama, Mistral y DeepSeek.
- **07 Aplicaciones Empresariales**: mapear casos reales de industria con arquitecturas Transformer.
- **08 Actividad Final**: construir un asistente documental integrador end-to-end.

## Notas didácticas

- Cada notebook es independiente: importa sus propias librerías y carga datos por rutas relativas.
- Se muestran resultados intermedios con `print()` o `display()`.
- Todos los notebooks incluyen preguntas de reflexión para discusión en clase.
- Los modelos preentrenados son versiones ligeras para facilitar ejecución en laptops.

## Prerrequisitos

Completar la **Clase 01: Text Mining y Embeddings** antes de iniciar esta sesión.

## Actividad final del curso — MiniChatGPT Lab

Proyecto integrador en notebook (misma estructura que `IMDb_NLP_Processor` de la Clase 1):

```bash
cd ../MiniChatGPT_Lab
pip install -r requirements.txt
jupyter notebook notebooks/actividad.ipynb
```

Construye un **Mini ChatGPT** con GPT-2: system prompt, historial, temperatura y context window. Ver rúbrica en [MiniChatGPT_Lab/README.md](../MiniChatGPT_Lab/README.md).
