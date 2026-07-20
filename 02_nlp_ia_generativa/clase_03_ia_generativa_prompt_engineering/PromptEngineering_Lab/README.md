# PromptEngineering Lab

Actividad final de la **Unidad 2 - Clase 3: IA Generativa y Prompt Engineering**.

## Descripción

Laboratorio práctico para **diseñar, probar y comparar** estrategias de prompting con modelos en español:

- `datificate/gpt2-small-spanish` — generación de texto
- `pysentimiento/robertuito-sentiment-analysis` — sentimiento

- Anatomía completa del prompt (rol, contexto, instrucción, restricciones, formato)
- Zero-shot, one-shot, few-shot
- Chain-of-Thought y role prompting
- Context prompting con documentos empresariales
- Comparación side-by-side con evaluación simple
- Caso integrador: asistente de soporte TechNova SaaS

> Todo ocurre en el notebook. No requiere API keys.

## Estructura

```text
PromptEngineering_Lab/
├── README.md
├── requirements.txt
├── generate_actividad.py
├── notebooks/
│   └── actividad.ipynb
├── src/
│   ├── prompt_builder.py
│   ├── strategies.py
│   ├── llm_client.py
│   ├── evaluator.py
│   └── utils.py
├── datasets/
└── images/
```

## Instalación

```bash
pip install -r requirements.txt
jupyter notebook notebooks/actividad.ipynb
```

La primera ejecución descarga modelos en español desde Hugging Face (~510 MB GPT-2 Spanish + ~435 MB Robertuito).

> **Nota:** Usamos los mismos modelos que la Clase 02 (carpeta CLASE). Los prompts y ejemplos están en **español**. El objetivo es dominar las **técnicas de prompting**, no igualar ChatGPT.

## Partes del notebook

| Parte | Contenido |
|---|---|
| 1 | Cargar GPT-2 Spanish y Robertuito |
| 2 | Diseñar prompts con anatomía completa |
| 3 | Zero-shot, one-shot y few-shot |
| 4 | Chain-of-Thought y role prompting |
| 5 | Context prompting con documentos |
| 6 | Comparar prompting vs modelo de sentimiento |
| 7 | Caso práctico: asistente de soporte empresarial |

## Entregables

1. Notebook `actividad.ipynb` ejecutado completo.
2. Capturas en `images/` (mínimo 4):
   - Comparación zero-shot vs few-shot
   - Chain-of-Thought en acción
   - Context prompting con anti-alucinación
   - Tabla comparativa de estrategias
3. Respuesta escrita (250-350 palabras): *¿Qué estrategia de prompting usarías en tu trabajo y por qué?*

## Rúbrica (100 puntos)

| Criterio | Puntos |
|---|---:|
| Notebook ejecutado sin errores | 15 |
| Prompts estructurados con anatomía completa | 15 |
| Experimentos zero/one/few-shot | 15 |
| Chain-of-Thought y role prompting | 15 |
| Context prompting con restricciones | 15 |
| Comparación de estrategias documentada | 15 |
| Caso TechNova integrador | 5 |
| Reflexión escrita | 5 |

## Punto extra (+10)

Diseña un sistema de 3 prompts encadenados (extraer → clasificar → responder) para un caso de tu industria y documenta los resultados con la función `score_response()`.

## Regenerar notebook

```bash
python generate_actividad.py
```
