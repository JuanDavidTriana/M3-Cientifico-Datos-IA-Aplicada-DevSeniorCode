#!/usr/bin/env python3
"""Genera los 8 notebooks de la Clase 02 sobre Transformers."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT / "notebooks"
DATASETS = "../datasets"


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {"language": "markdown"},
        "source": text.splitlines(keepends=True) or [""],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _cell_id(),
        "metadata": {"language": "python"},
        "outputs": [],
        "source": text.splitlines(keepends=True) or [""],
    }


def footer(
    resultados: str,
    conclusiones: str,
    ejercicios_guiados_md: str,
    ejercicios_guiados_code: str,
    ejercicios_propuestos: str,
    preguntas: str,
) -> list[dict]:
    """Secciones finales comunes exigidas en todos los notebooks."""
    cells = [
        md(f"## Resultados\n{resultados}"),
        md(f"## Conclusiones\n{conclusiones}"),
        md(f"## Ejercicios guiados resueltos\n{ejercicios_guiados_md}"),
        code(ejercicios_guiados_code),
        md(f"## Ejercicios propuestos\n{ejercicios_propuestos}"),
        md(f"## Preguntas de reflexion\n{preguntas}"),
    ]
    return cells


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def header(title: str, objetivos: str, intro: str) -> list[dict]:
    return [
        md(f"# {title}\n\n## Objetivos\n{objetivos}\n\n## Introduccion\n{intro}"),
    ]


def build_notebook_01() -> dict:
    cells = header(
        "Notebook 01 - Historia de los Transformers",
        "- Comprender la evolucion del NLP desde reglas hasta Transformers.\n"
        "- Identificar limitaciones de RNN/LSTM frente a secuencias largas.\n"
        "- Reconocer aportes clave del paper *Attention Is All You Need*.",
        "Antes de 2017, el NLP dependia de redes recurrentes que procesaban el texto "
        "token a token. Eso generaba cuellos de botella de memoria y dificultad para "
        "capturar dependencias lejanas. Este notebook recorre esa historia con datos "
        "y una linea de tiempo interactiva.",
    )
    cells += [
        code(
            "# Importamos librerias para tablas y graficos\n"
            "from pathlib import Path\n"
            "from IPython.display import display\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "# Definimos la ruta base del proyecto\n"
            "BASE = Path('..')\n"
            "print('Entorno listo para el notebook 01')"
        ),
        md(
            "## 1) Linea de tiempo del NLP\n"
            "La siguiente tabla resume hitos desde enfoques simbolicos hasta modelos "
            "fundacionales actuales."
        ),
        code(
            "# Construimos un DataFrame con eventos historicos relevantes\n"
            "eventos = [\n"
            "    {'anio': 1950, 'hito': 'Turing y computacion simbolica', 'paradigma': 'Reglas'},\n"
            "    {'anio': 1990, 'hito': 'Modelos estadisticos n-gram', 'paradigma': 'Estadistica'},\n"
            "    {'anio': 2010, 'hito': 'Word2Vec y embeddings densos', 'paradigma': 'Embeddings'},\n"
            "    {'anio': 2014, 'hito': 'Seq2Seq con LSTM + atencion', 'paradigma': 'RNN'},\n"
            "    {'anio': 2017, 'hito': 'Attention Is All You Need', 'paradigma': 'Transformer'},\n"
            "    {'anio': 2018, 'hito': 'BERT (encoder-only)', 'paradigma': 'Preentrenamiento'},\n"
            "    {'anio': 2019, 'hito': 'GPT-2 (decoder-only)', 'paradigma': 'Generacion'},\n"
            "    {'anio': 2020, 'hito': 'T5 texto-a-texto', 'paradigma': 'Unificacion'},\n"
            "    {'anio': 2023, 'hito': 'LLMs conversacionales a escala', 'paradigma': 'Fundacionales'},\n"
            "]\n"
            "df_eventos = pd.DataFrame(eventos)\n"
            "display(df_eventos)"
        ),
        md(
            "## 2) Problemas de RNN y LSTM\n"
            "Las redes recurrentes mantienen un estado oculto que se actualiza secuencialmente. "
            "Eso implica vanishing gradient, dificultad con dependencias largas y poca paralelizacion."
        ),
        code(
            "# Simulamos tiempos relativos de procesamiento por longitud de secuencia\n"
            "import numpy as np\n"
            "\n"
            "longitudes = np.array([10, 50, 100, 200, 500])\n"
            "# Tiempo proporcional a la longitud (procesamiento secuencial)\n"
            "tiempo_rnn = longitudes * 1.0\n"
            "# Transformer procesa en paralelo; el costo crece pero con mejor escalado\n"
            "tiempo_transformer = np.log(longitudes + 1) * 15\n"
            "\n"
            "comparacion = pd.DataFrame({\n"
            "    'longitud_tokens': longitudes,\n"
            "    'costo_relativo_rnn': tiempo_rnn,\n"
            "    'costo_relativo_transformer': tiempo_transformer.round(2),\n"
            "})\n"
            "display(comparacion)"
        ),
        md(
            "## 3) Datos clave del paper de 2017\n"
            "El articulo *Attention Is All You Need* (Vaswani et al.) propuso reemplazar "
            "recurrencia por mecanismos de atencion multi-cabeza."
        ),
        code(
            "# Recopilamos hechos tecnicos del paper original\n"
            "paper_facts = pd.DataFrame([\n"
            "    {'concepto': 'Arquitectura base', 'detalle': 'Encoder-Decoder con Self-Attention'},\n"
            "    {'concepto': 'Complejidad por capa', 'detalle': 'O(n^2 * d) respecto a longitud n'},\n"
            "    {'concepto': 'Positional Encoding', 'detalle': 'Senos y cosenos para orden'},\n"
            "    {'concepto': 'Multi-Head Attention', 'detalle': 'Varias cabezas en paralelo'},\n"
            "    {'concepto': 'BLEU WMT 2014 EN-DE', 'detalle': '28.4 (state of the art en su epoca)'},\n"
            "    {'concepto': 'Entrenamiento', 'detalle': 'Altamente paralelizable en GPU/TPU'},\n"
            "])\n"
            "display(paper_facts)"
        ),
        md("## 4) Visualizacion interactiva de la evolucion"),
        code(
            "# Graficamos la linea de tiempo con pandas y matplotlib\n"
            "fig, ax = plt.subplots(figsize=(12, 5))\n"
            "ax.scatter(df_eventos['anio'], df_eventos['paradigma'], s=120, c='steelblue')\n"
            "\n"
            "# Etiquetamos cada punto con el hito principal\n"
            "for _, row in df_eventos.iterrows():\n"
            "    ax.annotate(\n"
            "        f\"{row['anio']}: {row['hito']}\",\n"
            "        (row['anio'], row['paradigma']),\n"
            "        textcoords='offset points',\n"
            "        xytext=(0, 8),\n"
            "        ha='center',\n"
            "        fontsize=8,\n"
            "    )\n"
            "\n"
            "ax.set_xlabel('Anio')\n"
            "ax.set_ylabel('Paradigma dominante')\n"
            "ax.set_title('Evolucion historica del procesamiento de lenguaje natural')\n"
            "ax.grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    ]
    cells += footer(
        "Obtuvimos una tabla cronologica, una comparacion cualitativa RNN vs Transformer "
        "y un grafico de evolucion del paradigma dominante.",
        "Los Transformers surgieron como respuesta directa a limitaciones de RNN/LSTM. "
        "Su paralelizacion y atencion global cambiaron el ritmo de innovacion en NLP.",
        "**Ejercicio 1:** Agrega un evento de 2024 relacionado con modelos abiertos.\n\n"
        "**Solucion:**",
        "# Agregamos una fila nueva al DataFrame de eventos\n"
        "nuevo = pd.DataFrame([{\n"
        "    'anio': 2024,\n"
        "    'hito': 'Modelos open-weight de alta eficiencia',\n"
        "    'paradigma': 'Fundacionales',\n"
        "}])\n"
        "df_actualizado = pd.concat([df_eventos, nuevo], ignore_index=True)\n"
        "display(df_actualizado.tail(3))",
        "1. Investiga que problema resolvio Word2Vec que n-gram no resolvia bien.\n"
        "2. Compara BLEU de Seq2Seq clasicos vs Transformer en traduccion.\n"
        "3. Elabora un diagrama de arquitectura encoder-decoder sin usar recurrencia.",
        "1. Por que la atencion global es costosa en secuencias muy largas?\n"
        "2. Que ventaja aporta el preentrenamiento mas alla de la arquitectura?\n"
        "3. En que tareas aun pueden ser utiles modelos mas pequenos que un LLM?",
    )
    return notebook(cells)


def build_notebook_02() -> dict:
    qkv_frases = [
        "El gato persigue al raton",
        "María le dijo a Ana que ella ganó",
        "Los transformers usan atencion",
        "Python es un lenguaje de programacion",
        "La self-attention calcula pesos",
        "El perro ladra en el jardin",
        "OpenAI publico GPT",
        "BERT enmascara tokens aleatorios",
        "La traduccion automatica mejoro con atencion",
        "El token CLS resume la frase",
        "Los embeddings capturan semantica",
        "El modelo aprende relaciones contextuales",
    ]
    frases_literal = ",\n    ".join(f'"{f}"' for f in qkv_frases)

    cells = header(
        "Notebook 02 - Embeddings, Tokenizacion y Self-Attention",
        "- Usar tokenizadores de Hugging Face sobre texto real.\n"
        "- Relacionar tokens con vectores de embedding.\n"
        "- Implementar Scaled Dot-Product Attention desde cero con PyTorch.",
        "La Self-Attention es el nucleo del Transformer. Antes de cargar modelos "
        "preentrenados, conviene entender como se calculan Q, K y V y como se obtienen "
        "pesos de atencion normalizados.",
    )
    cells += [
        code(
            "# Importamos PyTorch y utilidades de tokenizacion\n"
            "from pathlib import Path\n"
            "from IPython.display import display\n"
            "import torch\n"
            "import torch.nn.functional as F\n"
            "import pandas as pd\n"
            "from transformers import AutoTokenizer\n"
            "\n"
            "print('PyTorch version:', torch.__version__)"
        ),
        md("## 1) Tokenizacion con Hugging Face"),
        code(
            "# Cargamos un tokenizador simple (sin modelo pesado)\n"
            "tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')\n"
            "\n"
            "frase_ejemplo = 'Los transformers revolucionaron el NLP'\n"
            "# Convertimos texto a IDs de tokens\n"
            "encoding = tokenizer(frase_ejemplo, return_tensors='pt')\n"
            "print('Tokens:', tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))\n"
            "print('input_ids shape:', encoding['input_ids'].shape)"
        ),
        md("## 2) Embeddings aleatorios para demostracion"),
        code(
            "# Definimos dimension de embedding y vocabulario pequeno\n"
            "vocab_size = tokenizer.vocab_size\n"
            "embed_dim = 64\n"
            "\n"
            "# Capa de embedding inicializada aleatoriamente (solo didactica)\n"
            "embedding_layer = torch.nn.Embedding(vocab_size, embed_dim)\n"
            "\n"
            "# Obtenemos vectores para la frase tokenizada\n"
            "embeds = embedding_layer(encoding['input_ids'])\n"
            "print('Shape embeddings:', embeds.shape)  # (batch, seq_len, embed_dim)"
        ),
        md("## 3) Proyecciones Q, K y V"),
        code(
            "# Creamos matrices de proyeccion lineales para Q, K y V\n"
            "d_model = embed_dim\n"
            "W_q = torch.nn.Linear(d_model, d_model, bias=False)\n"
            "W_k = torch.nn.Linear(d_model, d_model, bias=False)\n"
            "W_v = torch.nn.Linear(d_model, d_model, bias=False)\n"
            "\n"
            "# Calculamos consultas, claves y valores\n"
            "Q = W_q(embeds)\n"
            "K = W_k(embeds)\n"
            "V = W_v(embeds)\n"
            "print('Shapes Q/K/V:', Q.shape, K.shape, V.shape)"
        ),
        md("## 4) Scaled Dot-Product Attention desde cero"),
        code(
            "def scaled_dot_product_attention(Q, K, V, mask=None):\n"
            "    \"\"\"Implementacion didactica de atencion escalada.\"\"\"\n"
            "    # d_k es la dimension de las claves\n"
            "    d_k = K.size(-1)\n"
            "    # Puntajes = QK^T dividido por sqrt(d_k)\n"
            "    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)\n"
            "    if mask is not None:\n"
            "        scores = scores.masked_fill(mask == 0, float('-inf'))\n"
            "    # Softmax sobre la ultima dimension\n"
            "    weights = F.softmax(scores, dim=-1)\n"
            "    # Salida ponderada por V\n"
            "    output = torch.matmul(weights, V)\n"
            "    return output, weights\n"
            "\n"
            "salida, pesos = scaled_dot_product_attention(Q, K, V)\n"
            "print('Salida atencion shape:', salida.shape)\n"
            "print('Matriz de pesos shape:', pesos.shape)"
        ),
        md("## 5) Intuicion Q/K/V con 12 frases de ejemplo"),
        code(
            f"# Lista de frases para explorar tokenizacion y longitudes\n"
            f"frases_qkv = [\n"
            f"    {frases_literal}\n"
            f"]\n"
            "\n"
            "resumen = []\n"
            "for frase in frases_qkv:\n"
            "    ids = tokenizer(frase, return_tensors='pt')['input_ids']\n"
            "    tokens = tokenizer.convert_ids_to_tokens(ids[0])\n"
            "    resumen.append({\n"
            "        'frase': frase,\n"
            "        'num_tokens': len(tokens),\n"
            "        'tokens': ' '.join(tokens),\n"
            "    })\n"
            "\n"
            "df_qkv = pd.DataFrame(resumen)\n"
            "display(df_qkv)"
        ),
        md("## 6) Interpretacion de pesos de atencion en una frase"),
        code(
            "# Tomamos una frase corta y mostramos pesos token-token\n"
            "frase_corta = 'El gato persigue al raton'\n"
            "enc = tokenizer(frase_corta, return_tensors='pt')\n"
            "emb = embedding_layer(enc['input_ids'])\n"
            "q, k, v = W_q(emb), W_k(emb), W_v(emb)\n"
            "_, attn = scaled_dot_product_attention(q, k, v)\n"
            "\n"
            "tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])\n"
            "matriz = pd.DataFrame(attn[0].detach().numpy(), index=tokens, columns=tokens)\n"
            "display(matriz.round(3))"
        ),
    ]
    cells += footer(
        "Tokenizamos texto, generamos embeddings, proyectamos Q/K/V e implementamos "
        "atencion escalada sin modelos preentrenados.",
        "La Self-Attention permite que cada token consulte a todos los demas. "
        "Escalar por sqrt(d_k) estabiliza los gradientes cuando la dimension crece.",
        "**Ejercicio:** Calcula la suma por fila de la matriz de atencion y verifica que sea 1.\n\n"
        "**Solucion:**",
        "sumas_fila = attn[0].sum(dim=-1)\n"
        "print('Sumas por fila (deben ser ~1):', sumas_fila.detach().numpy().round(4))",
        "1. Implementa una mascara causal para un decoder.\n"
        "2. Compara BPE vs WordPiece en una misma frase.\n"
        "3. Visualiza heatmap de `matriz` con seaborn.",
        "1. Por que Q, K y V usan proyecciones distintas?\n"
        "2. Que ocurre si omites la division por sqrt(d_k)?\n"
        "3. Como cambia el numero de tokens al usar subpalabras?",
    )
    return notebook(cells)


def build_notebook_03() -> dict:
    cells = header(
        "Notebook 03 - Visualizacion de Atencion",
        "- Extraer pesos de atencion de DistilBERT.\n"
        "- Construir heatmaps interpretables.\n"
        "- Analizar frases con ambiguedad desde CSV.",
        "Ver la matriz de atencion ayuda a entender que tokens 'miran' a cuales. "
        "Usaremos `distilbert-base-uncased` con `output_attentions=True`.",
    )
    cells += [
        code(
            "from pathlib import Path\n"
            "from IPython.display import display\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import torch\n"
            "from transformers import AutoTokenizer, AutoModel\n"
            "\n"
            "RUTA_FRASES = Path('..') / 'datasets' / 'frases_atencion.csv'\n"
            "print('Ruta datasets:', RUTA_FRASES.exists())"
        ),
        md("## 1) Cargar frases de atencion"),
        code(
            "# Leemos el CSV con frases categorizadas\n"
            "df_frases = pd.read_csv(RUTA_FRASES)\n"
            "display(df_frases.head())\n"
            "print('Total frases:', len(df_frases))"
        ),
        md("## 2) Cargar DistilBERT con salida de atenciones"),
        code(
            "# Nombre del modelo ligero en ingles\n"
            "model_name = 'distilbert-base-uncased'\n"
            "tokenizer = AutoTokenizer.from_pretrained(model_name)\n"
            "model = AutoModel.from_pretrained(model_name, output_attentions=True)\n"
            "model.eval()  # Modo inferencia\n"
            "print('Modelo cargado:', model_name)"
        ),
        md("## 3) Funcion para obtener matriz de atencion"),
        code(
            "def obtener_atencion(frase: str, capa: int = -1, cabeza: int = 0):\n"
            "    \"\"\"Devuelve tokens y matriz de atencion de una cabeza especifica.\"\"\"\n"
            "    inputs = tokenizer(frase, return_tensors='pt')\n"
            "    with torch.no_grad():\n"
            "        outputs = model(**inputs)\n"
            "    # outputs.attentions es tupla por capa\n"
            "    attn = outputs.attentions[capa][0, cabeza].cpu().numpy()\n"
            "    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])\n"
            "    return tokens, attn\n"
            "\n"
            "tokens_demo, attn_demo = obtener_atencion('The cat chased the mouse')\n"
            "print('Tokens:', tokens_demo)\n"
            "print('Shape atencion:', attn_demo.shape)"
        ),
        md("## 4) Heatmap con matplotlib y seaborn"),
        code(
            "def plot_heatmap(tokens, matriz, titulo='Matriz de atencion'):\n"
            "    plt.figure(figsize=(8, 6))\n"
            "    sns.heatmap(\n"
            "        matriz,\n"
            "        xticklabels=tokens,\n"
            "        yticklabels=tokens,\n"
            "        cmap='viridis',\n"
            "        annot=False,\n"
            "    )\n"
            "    plt.title(titulo)\n"
            "    plt.xlabel('Key tokens')\n"
            "    plt.ylabel('Query tokens')\n"
            "    plt.xticks(rotation=45, ha='right')\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "\n"
            "plot_heatmap(tokens_demo, attn_demo, 'DistilBERT - frase demo')"
        ),
        md("## 5) Analisis por categoria del CSV"),
        code(
            "# Seleccionamos una frase por tipo de ambiguedad\n"
            "muestras = df_frases.groupby('nota', as_index=False).first()\n"
            "\n"
            "for _, row in muestras.iterrows():\n"
            "    frase = row['frase']\n"
            "    categoria = row['nota']\n"
            "    tokens, mat = obtener_atencion(frase)\n"
            "    plot_heatmap(tokens, mat, f\"{categoria}: {frase[:40]}...\")"
        ),
        md("## 6) Interpretacion cualitativa"),
        code(
            "# Calculamos que token recibe mas atencion promedio en cada fila\n"
            "promedio_por_query = attn_demo.mean(axis=1)\n"
            "interpretacion = pd.DataFrame({\n"
            "    'token_query': tokens_demo,\n"
            "    'atencion_promedio': promedio_por_query,\n"
            "})\n"
            "display(interpretacion.sort_values('atencion_promedio', ascending=False))"
        ),
    ]
    cells += footer(
        "Visualizamos matrices de atencion para frases del CSV y una demo en ingles, "
        "identificando patrones por capa y cabeza.",
        "La atencion no es explicabilidad perfecta, pero orienta sobre dependencias "
        "lexicas. DistilBERT ofrece un equilibrio util entre costo e interpretacion.",
        "**Ejercicio:** Compara cabeza 0 vs cabeza 1 en la misma frase.\n\n"
        "**Solucion:**",
        "t0, a0 = obtener_atencion('The lawyer signed the contract', cabeza=0)\n"
        "t1, a1 = obtener_atencion('The lawyer signed the contract', cabeza=1)\n"
        "print('Shapes:', a0.shape, a1.shape)\n"
        "print('Diferencia media abs:', abs(a0 - a1).mean())",
        "1. Promedia atencion de todas las cabezas de la ultima capa.\n"
        "2. Traduce frases del CSV y repite el analisis.\n"
        "3. Identifica tokens `[CLS]` y `[SEP]` en los heatmaps.",
        "1. Por que distintas cabezas pueden especializarse?\n"
        "2. Que riesgos hay al sobre-interpretar un heatmap?\n"
        "3. Como afecta la longitud de la frase a la legibilidad del mapa?",
    )
    return notebook(cells)


def build_notebook_04() -> dict:
    cells = header(
        "Notebook 04 - BERT y Aplicaciones",
        "- Clasificar sentimiento con DistilBERT fine-tuned.\n"
        "- Responder preguntas con pipeline de QA.\n"
        "- Extraer entidades (NER) y embeddings contextuales.",
        "BERT es encoder-only: excelente para entender texto. Exploraremos sentimiento, "
        "QA, NER y vectores `[CLS]` sobre datasets del curso.",
    )
    cells += [
        code(
            "from pathlib import Path\n"
            "from IPython.display import display\n"
            "import pandas as pd\n"
            "import torch\n"
            "from transformers import (\n"
            "    pipeline,\n"
            "    AutoTokenizer,\n"
            "    AutoModel,\n"
            ")\n"
            "\n"
            "RUTA_REVIEWS = Path('..') / 'datasets' / 'reviews_sentiment.csv'\n"
            "RUTA_QA = Path('..') / 'datasets' / 'documentos_qa.csv'\n"
            "print('Archivos listos:', RUTA_REVIEWS.exists(), RUTA_QA.exists())"
        ),
        md("## 1) Sentimiento con modelo fine-tuned SST-2"),
        code(
            "# Pipeline de clasificacion de sentimiento en ingles\n"
            "sentiment = pipeline(\n"
            "    'sentiment-analysis',\n"
            "    model='distilbert-base-uncased-finetuned-sst-2-english',\n"
            ")\n"
            "\n"
            "df_reviews = pd.read_csv(RUTA_REVIEWS)\n"
            "display(df_reviews.head())"
        ),
        code(
            "# Traducimos etiquetas al ingles para el modelo SST-2\n"
            "mapa_en = {\n"
            "    'Excelente producto, llegó rápido y funciona perfectamente.': 'Excellent product, arrived fast and works perfectly.',\n"
            "    'Muy mala calidad, se rompió al segundo día.': 'Very bad quality, broke on the second day.',\n"
            "    'El servicio al cliente fue amable y resolvió mi problema.': 'Customer service was friendly and solved my problem.',\n"
            "    'Demasiado caro para lo que ofrece.': 'Too expensive for what it offers.',\n"
            "    'La interfaz es intuitiva y fácil de usar.': 'The interface is intuitive and easy to use.',\n"
            "}\n"
            "\n"
            "resultados = []\n"
            "for _, row in df_reviews.iterrows():\n"
            "    texto_es = row['texto']\n"
            "    texto_en = mapa_en.get(texto_es, texto_es)\n"
            "    pred = sentiment(texto_en[:512])[0]\n"
            "    resultados.append({\n"
            "        'texto': texto_es,\n"
            "        'sentimiento_real': row['sentimiento'],\n"
            "        'label_modelo': pred['label'],\n"
            "        'score': round(pred['score'], 4),\n"
            "    })\n"
            "\n"
            "df_sent = pd.DataFrame(resultados)\n"
            "display(df_sent)"
        ),
        md("## 2) Question Answering sobre documentos"),
        code(
            "# Pipeline extractivo de respuestas\n"
            "qa = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')\n"
            "df_qa = pd.read_csv(RUTA_QA)\n"
            "display(df_qa.head())"
        ),
        code(
            "respuestas = []\n"
            "for _, row in df_qa.iterrows():\n"
            "    out = qa(question=row['pregunta'], context=row['contexto'])\n"
            "    respuestas.append({\n"
            "        'pregunta': row['pregunta'],\n"
            "        'respuesta_esperada': row['respuesta'],\n"
            "        'respuesta_modelo': out['answer'],\n"
            "        'score': round(out['score'], 4),\n"
            "    })\n"
            "\n"
            "display(pd.DataFrame(respuestas))"
        ),
        md("## 3) Named Entity Recognition (NER)"),
        code(
            "# Pipeline NER en ingles\n"
            "ner = pipeline('ner', grouped_entities=True)\n"
            "texto_ner = 'Guido van Rossum created Python at Google Brain in 1991.'\n"
            "entidades = ner(texto_ner)\n"
            "display(pd.DataFrame(entidades))"
        ),
        md("## 4) Extraccion de embeddings contextuales"),
        code(
            "tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')\n"
            "model = AutoModel.from_pretrained('distilbert-base-uncased')\n"
            "model.eval()\n"
            "\n"
            "frase = 'Transformers changed natural language processing'\n"
            "inputs = tokenizer(frase, return_tensors='pt')\n"
            "with torch.no_grad():\n"
            "    hidden = model(**inputs).last_hidden_state\n"
            "\n"
            "# Vector CLS (primer token)\n"
            "cls_vector = hidden[0, 0, :]\n"
            "print('Dimension embedding CLS:', cls_vector.shape)\n"
            "print('Primeros 5 valores:', cls_vector[:5].tolist())"
        ),
    ]
    cells += footer(
        "Clasificamos reseñas, respondimos preguntas extractivas, detectamos entidades "
        "y extrajimos un embedding `[CLS]` contextual.",
        "BERT destaca en comprension y tareas de entendimiento. Para produccion conviene "
        "validar idioma, calibrar umbrales y considerar modelos multilingues.",
        "**Ejercicio:** Calcula accuracy aproximada mapeando POSITIVE->positivo.\n\n"
        "**Solucion:**",
        "df_sent['pred_es'] = df_sent['label_modelo'].map({'POSITIVE': 'positivo', 'NEGATIVE': 'negativo'})\n"
        "acc = (df_sent['pred_es'] == df_sent['sentimiento_real']).mean()\n"
        "print(f'Accuracy aproximada: {acc:.2%}')",
        "1. Prueba un modelo multilingue para reseñas en espanol.\n"
        "2. Fine-tune ligero de sentimiento con `Trainer`.\n"
        "3. Compara similitud coseno entre embeddings de dos frases.",
        "1. Cuando falla QA extractivo aun con contexto correcto?\n"
        "2. Que diferencia hay entre NER pipeline y token classification?\n"
        "3. Por que `[CLS]` resume la secuencia en BERT?",
    )
    return notebook(cells)


def build_notebook_05() -> dict:
    cells = header(
        "Notebook 05 - GPT y Generacion de Texto",
        "- Generar texto con GPT-2.\n"
        "- Comparar temperatura, top_p y top_k.\n"
        "- Aplicar prompts de resumen y reescritura.",
        "GPT es decoder-only y autoregresivo: predice el siguiente token. "
        "Los parametros de muestreo controlan creatividad vs coherencia.",
    )
    cells += [
        code(
            "from transformers import pipeline, set_seed\n"
            "from IPython.display import display\n"
            "import pandas as pd\n"
            "\n"
            "set_seed(42)\n"
            "generator = pipeline('text-generation', model='gpt2')\n"
            "print('GPT-2 listo para generar texto')"
        ),
        md("## 1) Generacion basica"),
        code(
            "prompt = 'In machine learning, transformers'\n"
            "salida = generator(prompt, max_new_tokens=40, do_sample=False)\n"
            "print(salida[0]['generated_text'])"
        ),
        md("## 2) Comparacion de temperatura"),
        code(
            "temperaturas = [0.3, 0.7, 1.2]\n"
            "filas = []\n"
            "for temp in temperaturas:\n"
            "    out = generator(\n"
            "        'The future of NLP is',\n"
            "        max_new_tokens=25,\n"
            "        do_sample=True,\n"
            "        temperature=temp,\n"
            "    )\n"
            "    filas.append({'temperatura': temp, 'texto': out[0]['generated_text']})\n"
            "\n"
            "display(pd.DataFrame(filas))"
        ),
        md("## 3) Top-p (nucleus) vs Top-k"),
        code(
            "base = 'Artificial intelligence can help companies'\n"
            "configs = [\n"
            "    {'nombre': 'top_k=10', 'top_k': 10, 'top_p': 1.0},\n"
            "    {'nombre': 'top_p=0.9', 'top_k': 0, 'top_p': 0.9},\n"
            "    {'nombre': 'top_p=0.5', 'top_k': 0, 'top_p': 0.5},\n"
            "]\n"
            "comparacion = []\n"
            "for cfg in configs:\n"
            "    out = generator(\n"
            "        base,\n"
            "        max_new_tokens=30,\n"
            "        do_sample=True,\n"
            "        temperature=0.8,\n"
            "        top_k=cfg['top_k'],\n"
            "        top_p=cfg['top_p'],\n"
            "    )\n"
            "    comparacion.append({'config': cfg['nombre'], 'texto': out[0]['generated_text']})\n"
            "\n"
            "display(pd.DataFrame(comparacion))"
        ),
        md("## 4) Prompt de resumen"),
        code(
            "documento = (\n"
            "    'Transformers replaced recurrent layers with self-attention, '\n"
            "    'enabling parallel training and better long-range dependencies.'\n"
            ")\n"
            "prompt_resumen = f'Summarize in one sentence:\\n{documento}\\nSummary:'\n"
            "resumen = generator(prompt_resumen, max_new_tokens=35, temperature=0.5)[0]['generated_text']\n"
            "print(resumen)"
        ),
        md("## 5) Prompt de reescritura"),
        code(
            "texto_formal = 'The quarterly results exceeded expectations due to strong demand.'\n"
            "prompt_rewrite = (\n"
            "    f'Rewrite for a general audience:\\n{texto_formal}\\nRewritten:'\n"
            ")\n"
            "reescrito = generator(prompt_rewrite, max_new_tokens=40, temperature=0.7)[0]['generated_text']\n"
            "print(reescrito)"
        ),
        md("## 6) Ejemplos de prompt engineering"),
        code(
            "prompts = [\n"
            "    'List three benefits of transformers in NLP:',\n"
            "    'Explain self-attention to a beginner:',\n"
            "    'Complete: Attention is all you need because',\n"
            "]\n"
            "for p in prompts:\n"
            "    txt = generator(p, max_new_tokens=30, temperature=0.6)[0]['generated_text']\n"
            "    print('-' * 60)\n"
            "    print(txt)"
        ),
    ]
    cells += footer(
        "Generamos texto con distintos parametros y observamos trade-offs entre "
        "diversidad, fluidez y repeticion.",
        "GPT-2 es pequeno pero ilustra principios de LLMs. Temperatura alta aumenta "
        "creatividad; top_p restringe el nucleo probabilistico de tokens candidatos.",
        "**Ejercicio:** Genera tres veces con la misma semilla y compara estabilidad.\n\n"
        "**Solucion:**",
        "set_seed(123)\n"
        "a = generator('Data science', max_new_tokens=15, do_sample=True, temperature=0.7)[0]['generated_text']\n"
        "set_seed(123)\n"
        "b = generator('Data science', max_new_tokens=15, do_sample=True, temperature=0.7)[0]['generated_text']\n"
        "print('Iguales con misma semilla:', a == b)",
        "1. Diseña un prompt few-shot con 2 ejemplos de tono formal.\n"
        "2. Mide longitud promedio generada por configuracion.\n"
        "3. Prueba penalizacion de repeticion (`repetition_penalty`).",
        "1. Cuando conviene `do_sample=False`?\n"
        "2. Que riesgo tiene temperatura muy alta en produccion?\n"
        "3. Como evaluarias calidad de un resumen generado?",
    )
    return notebook(cells)


def build_notebook_06() -> dict:
    cells = header(
        "Notebook 06 - Comparacion de Modelos",
        "- Contrastar arquitecturas BERT, GPT, T5, Llama, Mistral y DeepSeek.\n"
        "- Organizar metadatos en tabla pandas.\n"
        "- Relacionar tipo de modelo con casos de uso.",
        "No es necesario cargar todos los modelos: usaremos metadatos publicos "
        "para comparar paradigmas encoder, decoder y encoder-decoder.",
    )
    cells += [
        code(
            "from IPython.display import display\n"
            "import pandas as pd\n"
            "\n"
            "print('Tabla comparativa basada en metadatos (sin descarga de pesos)')"
        ),
        md("## 1) Tabla comparativa principal"),
        code(
            "modelos = pd.DataFrame([\n"
            "    {'modelo': 'BERT', 'tipo': 'Encoder-only', 'pretraining': 'MLM + NSP', 'fortaleza': 'Comprension', 'ejemplo_uso': 'Clasificacion, NER, QA extractivo'},\n"
            "    {'modelo': 'GPT-2/3/4', 'tipo': 'Decoder-only', 'pretraining': 'CLM autoregresivo', 'fortaleza': 'Generacion', 'ejemplo_uso': 'Chatbots, redaccion, codigo'},\n"
            "    {'modelo': 'T5', 'tipo': 'Encoder-Decoder', 'pretraining': 'Span corruption', 'fortaleza': 'Text-to-text', 'ejemplo_uso': 'Traduccion, resumen, QA generativo'},\n"
            "    {'modelo': 'Llama', 'tipo': 'Decoder-only', 'pretraining': 'CLM a escala', 'fortaleza': 'Razonamiento general', 'ejemplo_uso': 'Asistentes open-weight'},\n"
            "    {'modelo': 'Mistral', 'tipo': 'Decoder-only', 'pretraining': 'CLM eficiente', 'fortaleza': 'Costo/latencia', 'ejemplo_uso': 'Despliegue empresarial'},\n"
            "    {'modelo': 'DeepSeek', 'tipo': 'Decoder-only', 'pretraining': 'CLM + RL/DPO', 'fortaleza': 'Codigo y razonamiento', 'ejemplo_uso': 'Copilotos tecnicos'},\n"
            "])\n"
            "display(modelos)"
        ),
        md("## 2) Parametros y contexto (aproximados)"),
        code(
            "escala = pd.DataFrame([\n"
            "    {'modelo': 'BERT-base', 'parametros': '110M', 'contexto': '512', 'open_weights': 'Si'},\n"
            "    {'modelo': 'GPT-2', 'parametros': '124M-1.5B', 'contexto': '1024', 'open_weights': 'Si'},\n"
            "    {'modelo': 'T5-base', 'parametros': '220M', 'contexto': '512', 'open_weights': 'Si'},\n"
            "    {'modelo': 'Llama 3', 'parametros': '8B-70B', 'contexto': '8192+', 'open_weights': 'Parcial'},\n"
            "    {'modelo': 'Mistral 7B', 'parametros': '7B', 'contexto': '8192+', 'open_weights': 'Si'},\n"
            "    {'modelo': 'DeepSeek Coder', 'parametros': '1.3B-33B', 'contexto': '4096+', 'open_weights': 'Si'},\n"
            "])\n"
            "display(escala)"
        ),
        md("## 3) Matriz modelo vs tarea"),
        code(
            "tareas = ['Clasificacion', 'NER', 'QA extractivo', 'Generacion libre', 'Traduccion', 'Codigo']\n"
            "matriz = pd.DataFrame({\n"
            "    'BERT': ['Alta', 'Alta', 'Alta', 'Baja', 'Media', 'Baja'],\n"
            "    'GPT': ['Media', 'Media', 'Media', 'Alta', 'Media', 'Alta'],\n"
            "    'T5': ['Alta', 'Alta', 'Alta', 'Alta', 'Alta', 'Media'],\n"
            "    'Llama/Mistral/DeepSeek': ['Media', 'Media', 'Media', 'Alta', 'Alta', 'Alta'],\n"
            "}, index=tareas)\n"
            "display(matriz)"
        ),
        md("## 4) Visualizacion rapida"),
        code(
            "import matplotlib.pyplot as plt\n"
            "\n"
            "conteo_tipo = modelos['tipo'].value_counts()\n"
            "plt.figure(figsize=(6, 4))\n"
            "conteo_tipo.plot(kind='bar', color='teal')\n"
            "plt.title('Distribucion de tipos de arquitectura en la muestra')\n"
            "plt.ylabel('Cantidad de modelos')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    ]
    cells += footer(
        "Construimos tablas comparativas de arquitectura, escala y adecuacion por tarea "
        "sin cargar checkpoints pesados.",
        "No existe un modelo universal: encoder-only entiende, decoder-only genera, "
        "encoder-decoder unifica tareas texto-a-texto. La eleccion depende de latencia, "
        "costo y gobernanza de datos.",
        "**Ejercicio:** Filtra modelos decoder-only.\n\n"
        "**Solucion:**",
        "decoder_only = modelos[modelos['tipo'] == 'Decoder-only']\n"
        "display(decoder_only[['modelo', 'fortaleza', 'ejemplo_uso']])",
        "1. Agrega columnas de licencia y fecha de lanzamiento.\n"
        "2. Investiga que es MoE y donde aparece (Mixtral).\n"
        "3. Propón stack hibrido BERT+GPT para un chatbot empresarial.",
        "1. Por que los LLMs decoder dominan interfaces conversacionales?\n"
        "2. Cuando T5 sigue siendo preferible?\n"
        "3. Que implica `open_weights: Parcial` para una empresa regulada?",
    )
    return notebook(cells)


def build_notebook_07() -> dict:
    cells = header(
        "Notebook 07 - Aplicaciones Empresariales",
        "- Analizar casos reales por sector desde CSV.\n"
        "- Visualizar distribucion de aplicaciones y arquitecturas.\n"
        "- Discutir 20+ ejemplos de empresas con Transformers.",
        "Las arquitecturas Transformer ya impactan finanzas, salud, retail y tecnologia. "
        "Mapearemos casos del dataset a tipos de modelo y valor de negocio.",
    )
    cells += [
        code(
            "from pathlib import Path\n"
            "from IPython.display import display\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "\n"
            "RUTA = Path('..') / 'datasets' / 'casos_empresariales.csv'\n"
            "df = pd.read_csv(RUTA)\n"
            "print('Empresas cargadas:', len(df))"
        ),
        md("## 1) Exploracion del dataset"),
        code(
            "display(df.head(10))\n"
            "print('Columnas:', df.columns.tolist())"
        ),
        md("## 2) Analisis por sector"),
        code(
            "por_sector = df.groupby('sector').size().reset_index(name='conteo')\n"
            "display(por_sector.sort_values('conteo', ascending=False))"
        ),
        md("## 3) Visualizaciones"),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "\n"
            "df['sector'].value_counts().plot(kind='bar', ax=axes[0], color='coral')\n"
            "axes[0].set_title('Casos por sector')\n"
            "axes[0].tick_params(axis='x', rotation=45)\n"
            "\n"
            "df['modelo_tipo'].value_counts().plot(kind='barh', ax=axes[1], color='slateblue')\n"
            "axes[1].set_title('Tipos de arquitectura usados')\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md("## 4) Mapeo arquitectura -> caso de uso"),
        code(
            "mapa = df.groupby(['modelo_tipo', 'aplicacion']).size().reset_index(name='frecuencia')\n"
            "display(mapa.sort_values('frecuencia', ascending=False))"
        ),
        md("## 5) Discusion: 20 empresas reales"),
        code(
            "# Mostramos todas las empresas con su valor de negocio\n"
            "for i, row in df.iterrows():\n"
            "    print(f\"{i+1:02d}. {row['empresa']} ({row['sector']})\")\n"
            "    print(f\"    Aplicacion: {row['aplicacion']}\")\n"
            "    print(f\"    Modelo: {row['modelo_tipo']}\")\n"
            "    print(f\"    Valor: {row['descripcion']}\")\n"
            "    print('-' * 70)"
        ),
        md("## 6) Heatmap sector vs tipo de modelo"),
        code(
            "pivot = pd.crosstab(df['sector'], df['modelo_tipo'])\n"
            "plt.figure(figsize=(10, 6))\n"
            "sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues')\n"
            "plt.title('Sectores vs arquitecturas Transformer')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
    ]
    cells += footer(
        "Analizamos 20 casos empresariales, visualizamos sectores y arquitecturas, "
        "y relacionamos aplicaciones con tipos de modelo.",
        "El valor empresarial no viene solo del tamano del modelo, sino de integrarlo "
        "en flujos con metricas, seguridad y supervision humana cuando aplica.",
        "**Ejercicio:** Encuentra el sector con mayor diversidad de arquitecturas.\n\n"
        "**Solucion:**",
        "diversidad = pivot.astype(bool).sum(axis=1).sort_values(ascending=False)\n"
        "print('Sector mas diverso:', diversidad.index[0], 'con', diversidad.iloc[0], 'tipos')",
        "1. Agrega 5 casos colombianos/latam con fuentes publicas.\n"
        "2. Estima costo computacional por tipo de arquitectura.\n"
        "3. Define KPIs para un chatbot de soporte en e-commerce.",
        "1. Que sectores priorizan encoder-only y por que?\n"
        "2. Como mitigar alucinaciones en finanzas o salud?\n"
        "3. Que datos internos NO deberian enviarse a APIs publicas?",
    )
    return notebook(cells)


def build_notebook_08() -> dict:
    cells = header(
        "Notebook 08 - Actividad Final: Asistente Documental Inteligente",
        "- Integrar sentimiento BERT, QA y resumen GPT.\n"
        "- Procesar `documentos_qa.csv` y textos personalizados.\n"
        "- Documentar configuracion, resultados y mejoras futuras.",
        "Proyecto integrador resuelto: un pipeline que analiza documentos, "
        "responde preguntas, estima tono y genera resumenes ejecutivos.",
    )
    cells += [
        code(
            "from pathlib import Path\n"
            "from dataclasses import dataclass\n"
            "from IPython.display import display\n"
            "import pandas as pd\n"
            "from transformers import pipeline, set_seed\n"
            "\n"
            "set_seed(42)\n"
            "RUTA_QA = Path('..') / 'datasets' / 'documentos_qa.csv'\n"
            "print('Iniciando Asistente Documental Inteligente')"
        ),
        md("## 1) Configuracion del pipeline"),
        code(
            "@dataclass\n"
            "class ConfigAsistente:\n"
            "    modelo_sentimiento: str = 'distilbert-base-uncased-finetuned-sst-2-english'\n"
            "    modelo_qa: str = 'distilbert-base-cased-distilled-squad'\n"
            "    modelo_generacion: str = 'gpt2'\n"
            "    max_tokens_resumen: int = 60\n"
            "    temperatura: float = 0.6\n"
            "\n"
            "cfg = ConfigAsistente()\n"
            "print(cfg)"
        ),
        md("## 2) Inicializar componentes"),
        code(
            "# Cargamos tres pipelines especializados\n"
            "sentiment_pipe = pipeline('sentiment-analysis', model=cfg.modelo_sentimiento)\n"
            "qa_pipe = pipeline('question-answering', model=cfg.modelo_qa)\n"
            "gen_pipe = pipeline('text-generation', model=cfg.modelo_generacion)\n"
            "print('Componentes listos: sentimiento, QA, generacion')"
        ),
        md("## 3) Funciones del asistente"),
        code(
            "def analizar_sentimiento(texto_en: str) -> dict:\n"
            "    out = sentiment_pipe(texto_en[:512])[0]\n"
            "    return {'label': out['label'], 'score': round(out['score'], 4)}\n"
            "\n"
            "def responder_pregunta(contexto: str, pregunta: str) -> dict:\n"
            "    out = qa_pipe(question=pregunta, context=contexto)\n"
            "    return {'answer': out['answer'], 'score': round(out['score'], 4)}\n"
            "\n"
            "def resumir_documento(texto: str) -> str:\n"
            "    prompt = f'Summarize clearly:\\n{texto}\\nSummary:'\n"
            "    out = gen_pipe(\n"
            "        prompt,\n"
            "        max_new_tokens=cfg.max_tokens_resumen,\n"
            "        do_sample=True,\n"
            "        temperature=cfg.temperatura,\n"
            "    )\n"
            "    return out[0]['generated_text']\n"
            "\n"
            "print('Funciones definidas correctamente')"
        ),
        md("## 4) Procesar documentos_qa.csv"),
        code(
            "df = pd.read_csv(RUTA_QA)\n"
            "resultados = []\n"
            "\n"
            "for _, row in df.iterrows():\n"
            "    contexto = row['contexto']\n"
            "    pregunta = row['pregunta']\n"
            "    qa = responder_pregunta(contexto, pregunta)\n"
            "    sent = analizar_sentimiento(contexto)\n"
            "    resumen = resumir_documento(contexto)\n"
            "    resultados.append({\n"
            "        'pregunta': pregunta,\n"
            "        'respuesta_esperada': row['respuesta'],\n"
            "        'respuesta_modelo': qa['answer'],\n"
            "        'qa_score': qa['score'],\n"
            "        'sentimiento': sent['label'],\n"
            "        'sentimiento_score': sent['score'],\n"
            "        'resumen': resumen,\n"
            "    })\n"
            "\n"
            "df_resultados = pd.DataFrame(resultados)\n"
            "display(df_resultados)"
        ),
        md("## 5) Texto personalizado del usuario"),
        code(
            "texto_custom = (\n"
            "    'Our support team resolved 95% of tickets within 24 hours. '\n"
            "    'Customers reported high satisfaction after the new chatbot launch.'\n"
            ")\n"
            "pregunta_custom = 'What percentage of tickets were resolved within 24 hours?'\n"
            "\n"
            "reporte = {\n"
            "    'sentimiento': analizar_sentimiento(texto_custom),\n"
            "    'respuesta': responder_pregunta(texto_custom, pregunta_custom),\n"
            "    'resumen': resumir_documento(texto_custom),\n"
            "}\n"
            "display(pd.DataFrame([reporte]))"
        ),
        md("## 6) Metricas agregadas del proyecto"),
        code(
            "aciertos_qa = (df_resultados['respuesta_modelo'].str.lower().str.strip()\n"
            "               == df_resultados['respuesta_esperada'].str.lower().str.strip())\n"
            "print('Exact match QA:', f\"{aciertos_qa.mean():.1%}\")\n"
            "print('Sentimiento positivo promedio:',\n"
            "      (df_resultados['sentimiento'] == 'POSITIVE').mean())"
        ),
        md(
            "## 7) Mejoras propuestas\n"
            "- Usar modelos multilingues para documentos en espanol.\n"
            "- Agregar chunking para contextos largos.\n"
            "- Evaluar con RAG sobre base vectorial corporativa.\n"
            "- Registrar trazas y scores en un dashboard de monitoreo."
        ),
    ]
    cells += footer(
        "El asistente combino tres capacidades sobre CSV y texto libre, con metricas "
        "basicas de exactitud QA y tono del documento.",
        "Un sistema documental util une comprension (BERT), extraccion (QA) y "
        "sintesis (GPT). La calidad mejora con datos del dominio y evaluacion continua.",
        "**Ejercicio:** Crea funcion `procesar_documento(texto, preguntas)` que devuelva JSON.\n\n"
        "**Solucion:**",
        "def procesar_documento(texto: str, preguntas: list[str]) -> dict:\n"
        "    return {\n"
        "        'sentimiento': analizar_sentimiento(texto),\n"
        "        'resumen': resumir_documento(texto),\n"
        "        'qa': [responder_pregunta(texto, p) for p in preguntas],\n"
        "    }\n"
        "\n"
        "ejemplo = procesar_documento(texto_custom, [pregunta_custom])\n"
        "print(list(ejemplo.keys()))",
        "1. Empaqueta el asistente como API FastAPI.\n"
        "2. Agrega validacion de entrada y limites de tokens.\n"
        "3. Disena pruebas unitarias para `responder_pregunta`.",
        "1. Que componente fallaria primero en documentos de 50 paginas?\n"
        "2. Como combinarias este pipeline con embeddings de Clase 01?\n"
        "3. Que metricas de negocio reportarias ademas de exact match?",
    )
    return notebook(cells)


NOTEBOOK_BUILDERS = {
    "01_Historia_Transformers.ipynb": build_notebook_01,
    "02_Embeddings_Tokenizacion_SelfAttention.ipynb": build_notebook_02,
    "03_Visualizacion_Atencion.ipynb": build_notebook_03,
    "04_BERT_Aplicaciones.ipynb": build_notebook_04,
    "05_GPT_Generacion_Texto.ipynb": build_notebook_05,
    "06_Comparacion_Modelos.ipynb": build_notebook_06,
    "07_Aplicaciones_Empresariales.ipynb": build_notebook_07,
    "08_Actividad_Final.ipynb": build_notebook_08,
}


def write_notebook(path: Path, nb: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=4)
        f.write("\n")


def main() -> list[str]:
    created: list[str] = []
    for filename, builder in NOTEBOOK_BUILDERS.items():
        out = NOTEBOOKS_DIR / filename
        write_notebook(out, builder())
        created.append(str(out))
        print(f"Generado: {out}")
    return created


if __name__ == "__main__":
    files = main()
    print(f"\nTotal notebooks generados: {len(files)}")
