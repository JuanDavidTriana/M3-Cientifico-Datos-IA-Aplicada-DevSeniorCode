# MiniChatGPT Lab

Actividad final de la **Unidad 2 - Clase 2: La Revolución de los Transformers**.

## Descripción

Construye tu propio **Mini ChatGPT** dentro de un notebook Jupyter, usando **GPT-2** (arquitectura decoder-only) y los mismos conceptos que explican ChatGPT, Claude o Gemini:

- System prompt
- Historial de conversación
- Predicción autoregresiva (siguiente token)
- Temperatura, top-p y top-k
- Context window (límite de tokens)

> Todo ocurre en el notebook. No hay frontend web separado.

## Estructura

```text
MiniChatGPT_Lab/
├── README.md
├── requirements.txt
├── notebooks/
│   └── actividad.ipynb
├── src/
│   ├── utils.py
│   ├── prompt_utils.py
│   ├── chat_engine.py
│   └── viz_utils.py
└── images/
```

## Instalación

```bash
pip install -r requirements.txt
jupyter notebook notebooks/actividad.ipynb
```

La primera ejecución descarga GPT-2 (~500 MB) desde Hugging Face.

> **Importante:** GPT-2 no es ChatGPT. Fue entrenado para continuar texto en inglés, sin instrucciones ni RLHF. Las demos del notebook usan preguntas en **inglés** para obtener respuestas más coherentes.

## Partes del notebook

| Parte | Contenido |
|---|---|
| 1 | Cargar GPT-2 y entender decoder-only |
| 2 | Primer mensaje: un turno de chat |
| 3 | Conversación multi-turno con historial |
| 4 | System prompt: cambiar personalidad del bot |
| 5 | Temperatura y top-p: comparar respuestas |
| 6 | Context window: ver cuándo se llena |
| 7 | Sesión libre: escribe tus propias preguntas |

## Entregables

1. Notebook `actividad.ipynb` ejecutado completo.
2. Capturas en `images/` (mínimo 3).
3. Respuesta escrita (200-300 palabras): *¿En qué se parece y en qué difiere tu Mini ChatGPT de ChatGPT real?*

## Rúbrica (100 puntos)

| Criterio | Puntos |
|---|---:|
| Notebook ejecutado sin errores | 20 |
| Conversación multi-turno funcional | 20 |
| Experimentos con system prompt | 15 |
| Comparación temperatura / top-p | 20 |
| Análisis de context window | 10 |
| Reflexión escrita | 10 |
| Código comentado y ordenado | 5 |

## Punto extra (+10)

Modificar `max_new_tokens`, probar 3 system prompts distintos (tutor, chef, programador) y documentar cómo cambia el estilo de respuesta.
