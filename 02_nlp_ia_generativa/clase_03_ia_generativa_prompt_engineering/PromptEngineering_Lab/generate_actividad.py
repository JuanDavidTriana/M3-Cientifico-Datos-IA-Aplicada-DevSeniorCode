"""Genera notebooks/actividad.ipynb para PromptEngineering Lab (modelos en español)."""

import json
from pathlib import Path

OUT = Path(__file__).parent / "notebooks" / "actividad.ipynb"

MODEL_GENERACION = "datificate/gpt2-small-spanish"
MODEL_SENTIMIENTO = "pysentimiento/robertuito-sentiment-analysis"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": [t]}


def code(t):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [t],
        "outputs": [],
        "execution_count": None,
    }


cells = [
    md(
        "# PromptEngineering Lab\n\n"
        "## Unidad 2 - Clase 3: IA Generativa y Prompt Engineering\n\n"
        "Laboratorio practico con **modelos en español** (misma configuracion que Clase 02):\n\n"
        f"- Generacion: `{MODEL_GENERACION}`\n"
        f"- Sentimiento: `{MODEL_SENTIMIENTO}`\n\n"
        "Tecnicas: anatomia del prompt, zero/one/few-shot, CoT, role y context prompting.\n\n"
        "> Ejecuta **Run All**. Si cambias archivos en `src/`, reinicia kernel."
    ),
    md(
        "## Objetivos\n"
        "1. Construir prompts estructurados en español.\n"
        "2. Aplicar 6 tecnicas de prompting con GPT-2 Spanish.\n"
        "3. Comparar zero-shot vs modelo de sentimiento Robertuito.\n"
        "4. Resolver el caso TechNova SaaS."
    ),
    md(
        "## Nota sobre los modelos\n\n"
        "Usamos los mismos modelos que en la Clase 02 (carpeta CLASE). "
        "GPT-2 Spanish fue entrenado para **continuar texto** en español. "
        "El objetivo es dominar las **tecnicas de prompting**, no igualar ChatGPT."
    ),
    code(
        "%matplotlib inline\n\n"
        "import importlib\n"
        "import sys\n"
        "import warnings\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "from IPython.display import Markdown, display\n"
        "from transformers import pipeline\n\n"
        "warnings.filterwarnings('ignore', category=UserWarning)\n\n"
        "def find_root(start: Path) -> Path:\n"
        "    for folder in [start, *start.parents]:\n"
        "        if (folder / 'src').exists() and 'PromptEngineering' in folder.name:\n"
        "            return folder\n"
        "    raise FileNotFoundError('Abre el notebook desde PromptEngineering_Lab/notebooks/')\n\n"
        "ROOT = find_root(Path.cwd())\n"
        "SRC = ROOT / 'src'\n"
        "IMAGES = ROOT / 'images'\n"
        "IMAGES.mkdir(parents=True, exist_ok=True)\n\n"
        "if str(SRC) not in sys.path:\n"
        "    sys.path.insert(0, str(SRC))\n\n"
        "import prompt_builder\n"
        "import strategies\n"
        "import llm_client\n"
        "import evaluator\n"
        "import utils\n\n"
        "importlib.reload(prompt_builder)\n"
        "importlib.reload(strategies)\n"
        "importlib.reload(llm_client)\n"
        "importlib.reload(evaluator)\n"
        "importlib.reload(utils)\n\n"
        "from prompt_builder import build_structured_prompt, STANDARD_CONSTRAINTS\n"
        "from strategies import zero_shot, one_shot, few_shot_from_dataframe, chain_of_thought, role_prompt, context_prompt\n"
        "from llm_client import LLMClient, LLMConfig, MODEL_GENERACION, MODEL_SENTIMIENTO\n"
        "from evaluator import score_response, compare_strategies\n"
        "from utils import load_dataset\n\n"
        "print('Entorno listo. ROOT =', ROOT)"
    ),
    md("# Parte 1 — Cargar modelos en español"),
    code(
        "config = LLMConfig(model_name=MODEL_GENERACION, max_new_tokens=50, temperature=0.7, seed=42)\n"
        "client = LLMClient(config=config)\n"
        "print('Cargando GPT-2 en español...')\n"
        "client.load()\n"
        "print('Cargando Robertuito (sentimiento)...')\n"
        f"sentimientos = pipeline('sentiment-analysis', model='{MODEL_SENTIMIENTO}')\n"
        "print('Modelos listos.')"
    ),
    md("# Parte 2 — Anatomia del prompt en español"),
    code(
        "prompt = build_structured_prompt(\n"
        "    role='Eres un analista de soporte al cliente.',\n"
        "    instruction='Clasifica el sentimiento como positivo, negativo o neutral.',\n"
        "    constraints='Responde SOLO con la etiqueta. Sin explicacion.',\n"
        "    output_format='Una sola palabra',\n"
        "    user_input='El producto llegó roto y nadie respondió a mis correos.',\n"
        ")\n"
        "print(prompt)\n"
        "print('\\n--- Respuesta ---')\n"
        "print(client.generate_completion(prompt, max_new_tokens=10, temperature=0.3))"
    ),
    md("# Parte 3 — Zero-shot, one-shot y few-shot"),
    code(
        "df_tickets = load_dataset('tickets_soporte.csv', ROOT)\n"
        "display(df_tickets.head())\n\n"
        "ticket_test = 'El sistema de pagos está caído no puedo procesar reembolsos'\n\n"
        "p_zero = zero_shot('Clasifica urgencia: alta, media, baja. Solo etiqueta:', ticket_test)\n"
        "p_one = one_shot('Clasifica urgencia.', 'Sistema caido', 'alta', ticket_test)\n"
        "p_few = few_shot_from_dataframe(\n"
        "    'Clasifica urgencia: alta, media, baja. Solo etiqueta:',\n"
        "    df_tickets, 'ticket', 'urgencia', ticket_test, n_examples=3,\n"
        ")\n\n"
        "comparacion = []\n"
        "for nombre, p in [('zero-shot', p_zero), ('one-shot', p_one), ('few-shot', p_few)]:\n"
        "    resp = client.generate_completion(p, max_new_tokens=5, temperature=0.3)\n"
        "    comparacion.append({'estrategia': nombre, 'respuesta': resp})\n"
        "display(pd.DataFrame(comparacion))"
    ),
    md("# Parte 4 — Chain-of-Thought y role prompting"),
    code(
        "problema = 'Una tienda tiene 23 manzanas. Usan 20 para pasteles y compran 6 más. ¿Cuántas manzanas tienen?'\n"
        "p_cot = chain_of_thought(problema)\n"
        "print('=== CoT ===')\n"
        "print(client.generate_completion(p_cot, max_new_tokens=60, temperature=0.5))\n\n"
        "roles = [\n"
        "    ('un profesor paciente para principiantes', '¿Qué es el aprendizaje automático?'),\n"
        "    ('un abogado laboral', '¿Qué es una cláusula de no competencia?'),\n"
        "]\n"
        "for rol, q in roles:\n"
        "    p = role_prompt(rol, q)\n"
        "    print(f'\\n=== Rol: {rol[:30]}... ===')\n"
        "    print(client.generate_completion(p, max_new_tokens=40, temperature=0.7))"
    ),
    md("# Parte 5 — Context prompting"),
    code(
        "df_docs = load_dataset('documentos_empresa.csv', ROOT)\n"
        "doc = df_docs[df_docs['titulo'] == 'Politica de devoluciones'].iloc[0]\n\n"
        "preguntas = [\n"
        "    '¿Puedo devolver un producto después de 25 días?',\n"
        "    '¿Cuál es el salario del CEO?',\n"
        "]\n"
        "for q in preguntas:\n"
        "    p = context_prompt(doc['contenido'], q, safe=True)\n"
        "    resp = client.generate_completion(p, max_new_tokens=30, temperature=0.3)\n"
        "    print(f'P: {q}')\n"
        "    print(f'R: {resp}')\n"
        "    print('-' * 50)"
    ),
    md("# Parte 6 — Comparar prompting vs modelo de sentimiento"),
    code(
        "df_reviews = load_dataset('reviews_sentiment.csv', ROOT)\n"
        "fila = df_reviews.iloc[0]\n"
        "texto = fila['texto']\n"
        "\n"
        "p_zs = zero_shot('Clasifica sentimiento positivo/negativo. Solo etiqueta:', texto)\n"
        "r_prompt = client.generate_completion(p_zs, max_new_tokens=5, temperature=0.3)\n"
        "r_modelo = sentimientos(texto)[0]\n"
        "\n"
        "display(pd.DataFrame([{\n"
        "    'texto': texto[:50],\n"
        "    'sentimiento_real': fila['sentimiento'],\n"
        "    'zero_shot_gpt': r_prompt,\n"
        "    'robertuito': r_modelo['label'],\n"
        "    'score': round(r_modelo['score'], 3),\n"
        "}]))\n\n"
        "resultados = []\n"
        "for nombre, p in [('zero-shot', p_zero), ('few-shot', p_few), ('cot', p_cot)]:\n"
        "    resp = client.generate_completion(p, max_new_tokens=30, temperature=0.3)\n"
        "    resultados.append({'estrategia': nombre, 'respuesta': resp, 'keywords': ['alta', 'media', 'baja']})\n"
        "evaluados = compare_strategies(resultados)\n"
        "df_eval = pd.DataFrame(evaluados)\n"
        "display(df_eval[['estrategia', 'has_content', 'total']])"
    ),
    md("# Parte 7 — Caso TechNova SaaS"),
    code(
        "def clasificar(ticket):\n"
        "    p = few_shot_from_dataframe(\n"
        "        'Clasifica urgencia: alta, media, baja. Solo etiqueta:',\n"
        "        df_tickets, 'ticket', 'urgencia', ticket, n_examples=4,\n"
        "    )\n"
        "    return client.generate_completion(p, max_new_tokens=5, temperature=0.3)\n\n"
        "def responder(pregunta):\n"
        "    ctx = '\\n'.join(f\"{r['titulo']}: {r['contenido']}\" for _, r in df_docs.iterrows())\n"
        "    p = build_structured_prompt(\n"
        "        role='Eres asistente de soporte de TechNova.',\n"
        "        context=ctx,\n"
        "        instruction='Responde la pregunta del cliente.',\n"
        "        constraints=STANDARD_CONSTRAINTS,\n"
        "        output_format='Maximo 2 oraciones',\n"
        "        user_input=pregunta,\n"
        "    )\n"
        "    return client.generate_completion(p, max_new_tokens=40, temperature=0.3)\n\n"
        "demo = pd.DataFrame([\n"
        "    {'funcion': 'clasificar', 'entrada': 'La app se cierra al abrir PDFs', 'salida': clasificar('La app se cierra al abrir PDFs')},\n"
        "    {'funcion': 'responder', 'entrada': '¿Cuál es la política de devoluciones?', 'salida': responder('¿Cuál es la política de devoluciones?')},\n"
        "    {'funcion': 'clasificar', 'entrada': '¿Cómo cambio mi contraseña?', 'salida': clasificar('¿Cómo cambio mi contraseña?')},\n"
        "])\n"
        "display(demo)"
    ),
    md(
        "## Entregables\n"
        "1. Notebook ejecutado completo.\n"
        "2. Capturas en `images/` (minimo 4).\n"
        "3. Reflexion (250-350 palabras): estrategia que usarias en tu trabajo.\n\n"
        "## Reflexion\n"
        "*Escribe aqui tu respuesta sobre que estrategia de prompting usarias en tu trabajo y por que.*"
    ),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Generado: {OUT}")
