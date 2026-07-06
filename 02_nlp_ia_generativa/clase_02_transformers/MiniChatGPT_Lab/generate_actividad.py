"""Genera notebooks/actividad.ipynb para MiniChatGPT Lab."""

import json
from pathlib import Path

OUT = Path(__file__).parent / "notebooks" / "actividad.ipynb"


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
        "# MiniChatGPT Lab\n\n"
        "## Unidad 2 - Clase 2: La Revolucion de los Transformers\n\n"
        "Construiras un **Mini ChatGPT** dentro de este notebook con GPT-2 (decoder-only):\n\n"
        "- System prompt\n"
        "- Historial de conversacion\n"
        "- Generacion autoregresiva\n"
        "- Temperatura, top-p y context window\n\n"
        "> Ejecuta **Run All** desde arriba. Si cambias archivos en `src/`, reinicia kernel y vuelve a ejecutar."
    ),
    md(
        "## Objetivos\n"
        "1. Cargar GPT-2 con Hugging Face Transformers.\n"
        "2. Enviar mensajes y recibir respuestas.\n"
        "3. Mantener historial multi-turno.\n"
        "4. Experimentar con system prompt y muestreo.\n"
        "5. Observar el limite de context window."
    ),
    md(
        "## Nota importante (GPT-2 vs ChatGPT)\n\n"
        "GPT-2 **no** es ChatGPT: no tiene instrucciones ni RLHF. Fue entrenado para **continuar texto** en ingles.\n\n"
        "- Usa preguntas en **ingles** en este notebook.\n"
        "- El system prompt orienta el estilo, pero no garantiza obediencia perfecta.\n"
        "- El objetivo es entender el **mecanismo**, no igualar ChatGPT comercial."
    ),
    code(
        "%matplotlib inline\n\n"
        "import importlib\n"
        "import sys\n"
        "import warnings\n"
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "from IPython.display import Markdown, display\n\n"
        "warnings.filterwarnings('ignore', category=UserWarning)\n\n"
        "def find_root(start: Path) -> Path:\n"
        "    for folder in [start, *start.parents]:\n"
        "        if (folder / 'src').exists() and 'MiniChatGPT' in folder.name:\n"
        "            return folder\n"
        "    raise FileNotFoundError('Abre el notebook desde MiniChatGPT_Lab/notebooks/')\n\n"
        "ROOT = find_root(Path.cwd())\n"
        "SRC = ROOT / 'src'\n"
        "IMAGES = ROOT / 'images'\n"
        "IMAGES.mkdir(parents=True, exist_ok=True)\n\n"
        "if str(SRC) not in sys.path:\n"
        "    sys.path.insert(0, str(SRC))\n\n"
        "import chat_engine\n"
        "import prompt_utils\n"
        "import viz_utils\n\n"
        "importlib.reload(prompt_utils)\n"
        "importlib.reload(chat_engine)\n"
        "importlib.reload(viz_utils)\n\n"
        "from chat_engine import ChatConfig, MiniChatGPT\n"
        "from viz_utils import plot_context_usage, plot_temperature_comparison\n\n"
        "print('Entorno listo.')\n"
        "print('ROOT =', ROOT)\n"
        "print('Device se detectara al cargar el modelo.')"
    ),
    md("# Parte 1 — Cargar el modelo\nGPT-2 es **decoder-only**: predice la siguiente palabra dado todo lo anterior."),
    code(
        "config = ChatConfig(\n"
        "    model_name='gpt2',\n"
        "    max_new_tokens=50,\n"
        "    temperature=0.8,\n"
        "    top_p=0.92,\n"
        "    top_k=50,\n"
        "    repetition_penalty=1.15,\n"
        "    max_context_tokens=512,\n"
        ")\n\n"
        "chat = MiniChatGPT(\n"
        "    config=config,\n"
        "    system_prompt=(\n"
        "        'You are a helpful AI tutor. '\n"
        "        'Answer in 1-2 clear sentences about NLP and Transformers.'\n"
        "    ),\n"
        ")\n\n"
        "print('Cargando GPT-2 (primera vez puede tardar)...')\n"
        "chat.load()\n"
        "print('MiniChatGPT listo en device:', chat._device)"
    ),
    md("# Parte 2 — Primer turno de chat\nUn turno = User + Assistant."),
    code(
        "chat.reset()\n\n"
        "pregunta_1 = 'What is a Transformer in machine learning?'\n"
        "resultado_1 = chat.chat(pregunta_1)\n\n"
        "print('Usuario:', resultado_1['user'])\n"
        "print('Asistente:', resultado_1['assistant'])\n"
        "print('Tokens entrada:', resultado_1['tokens_in'], '| salida:', resultado_1['tokens_out'])"
    ),
    code(
        "print('--- Prompt completo que vio el modelo ---')\n"
        "print(resultado_1['prompt'])"
    ),
    md("# Parte 3 — Conversacion multi-turno\nEl historial se acumula como en ChatGPT."),
    code(
        "chat.reset()\n\n"
        "turnos = [\n"
        "    'Explain self-attention in simple words.',\n"
        "    'Give me a one-sentence example.',\n"
        "    'Why is it better than LSTM for long text?',\n"
        "]\n\n"
        "for pregunta in turnos:\n"
        "    r = chat.chat(pregunta)\n"
        "    print('User:', r['user'])\n"
        "    print('Assistant:', r['assistant'])\n"
        "    print('-' * 60)\n\n"
        "display(chat.history_dataframe())"
    ),
    md("# Parte 4 — System prompt\nCambia la personalidad del asistente."),
    code(
        "chat.reset()\n"
        "chat.system_prompt = (\n"
        "    'You are a patient AI professor. Explain with simple analogies in 2 sentences.'\n"
        ")\n\n"
        "r_profesor = chat.chat('What is a token in NLP?')\n"
        "print('System prompt = profesor:')\n"
        "print(r_profesor['assistant'])"
    ),
    code(
        "chat.reset()\n"
        "chat.system_prompt = 'You are a concise technical assistant. Maximum 2 short sentences.'\n\n"
        "r_tecnico = chat.chat('What is a token in NLP?')\n"
        "print('System prompt = tecnico:')\n"
        "print(r_tecnico['assistant'])"
    ),
    md("# Parte 5 — Temperatura\nMisma pregunta, distinta creatividad."),
    code(
        "pregunta_temp = 'Explain why Transformers replaced LSTM in NLP.'\n"
        "temperaturas = [0.3, 0.8, 1.2]\n"
        "comparacion = []\n\n"
        "for temp in temperaturas:\n"
        "    chat.reset()\n"
        "    chat.config.temperature = temp\n"
        "    r = chat.chat(pregunta_temp)\n"
        "    comparacion.append({'temperature': temp, 'reply': r['assistant']})\n"
        "    print(f\"T={temp} -> {r['assistant']}\\n\")\n\n"
        "plot_temperature_comparison(\n"
        "    comparacion,\n"
        "    save_path=str(IMAGES / 'temperatura_comparacion.png'),\n"
        ")"
    ),
    md("# Parte 6 — Context window\nSi el historial crece, el contexto se satura."),
    code(
        "chat.reset()\n"
        "chat.system_prompt = 'You are a helpful assistant. Answer briefly.'\n\n"
        "for i in range(6):\n"
        "    chat.chat(f'Tell me fact number {i + 1} about neural networks.')\n\n"
        "tokens_usados = chat.context_token_count('')\n"
        "print('Tokens en historial:', tokens_usados)\n"
        "print('Limite configurado:', chat.config.max_context_tokens)\n\n"
        "plot_context_usage(\n"
        "    tokens_usados,\n"
        "    chat.config.max_context_tokens,\n"
        "    save_path=str(IMAGES / 'context_window.png'),\n"
        ")"
    ),
    md(
        "# Parte 7 — Tu sesion libre\n"
        "Edita `mis_preguntas` y ejecuta. Usa **ingles** para mejores resultados."
    ),
    code(
        "chat.reset()\n"
        "chat.system_prompt = (\n"
        "    'You are MiniChatGPT, an educational NLP assistant. '\n"
        "    'Answer clearly in 1-2 sentences.'\n"
        ")\n"
        "chat.config.temperature = 0.75\n"
        "chat.config.top_p = 0.9\n\n"
        "mis_preguntas = [\n"
        "    'What is self-attention?',\n"
        "    'Give an example with the ambiguous word bank.',\n"
        "    'What is the difference between BERT and GPT?',\n"
        "]\n\n"
        "print('=' * 60)\n"
        "print('MINI CHATGPT — SESION PERSONAL')\n"
        "print('=' * 60)\n\n"
        "for p in mis_preguntas:\n"
        "    r = chat.chat(p)\n"
        "    print(f\"\\nTu: {r['user']}\")\n"
        "    print(f\"Bot: {r['assistant']}\")\n"
        "    print(f\"(tokens in: {r['tokens_in']} | out: {r['tokens_out']})\")\n\n"
        "display(Markdown('### Historial completo'))\n"
        "display(chat.history_dataframe())"
    ),
    md(
        "## Cierre pedagogico\n\n"
        "### Preguntas de reflexion\n"
        "1. Que parte del flujo de ChatGPT replica tu Mini ChatGPT?\n"
        "2. Para que sirve el system prompt?\n"
        "3. Que cambio al subir la temperatura a 1.2?\n"
        "4. Por que existe un limite de context window?\n"
        "5. Que necesitarias para acercarte a ChatGPT real?\n\n"
        "### Entregable\n"
        "Compara tu Mini ChatGPT con ChatGPT comercial (200-300 palabras)."
    ),
    md(
        "## Ejercicio extra (+10 puntos)\n"
        "Prueba 3 system prompts (tutor, chef, programador) con la misma pregunta y guarda capturas en `images/`."
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Generado:", OUT)
