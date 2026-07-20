# Clase 03: IA Generativa y Prompt Engineering

Repositorio de apoyo para una clase de aproximadamente 4-6 horas del módulo **Científico de Datos e Inteligencia Artificial Aplicada**.

## Estructura del proyecto

```text
Clase-03-IA-Generativa-y-Prompt-Engineering/
│
├── README.md
├── requirements.txt
├── generate_notebooks.py
├── datasets/
│   ├── tareas_comparacion.csv
│   ├── prompts_buenos_malos.csv
│   ├── tickets_soporte.csv
│   ├── documentos_empresa.csv
│   ├── reviews_sentiment.csv
│   └── casos_practicos.csv
└── notebooks/
    ├── 01_IA_Generativa_y_LLMs.ipynb
    ├── 02_Anatomia_del_Prompt.ipynb
    ├── 03_Tecnicas_Prompting.ipynb
    ├── 04_Optimizacion_Prompts.ipynb
    ├── 05_Aplicaciones_Practicas.ipynb
    ├── 06_Comparacion_Estrategias.ipynb
    └── 07_Laboratorio_Final.ipynb
```

## Instalación

```bash
pip install -r requirements.txt
```

> **Nota:** La primera ejecución descargará modelos en español desde Hugging Face:
> - `datificate/gpt2-small-spanish` (~510 MB)
> - `pysentimiento/robertuito-sentiment-analysis` (~435 MB)
> - `mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es` (QA, notebooks 05 y 07)

## Cómo ejecutar los notebooks

```bash
jupyter notebook
```

Abrir la carpeta `notebooks/` y ejecutar cada notebook en orden.

## Orden recomendado

1. `01_IA_Generativa_y_LLMs.ipynb` — IA tradicional vs generativa, LLMs
2. `02_Anatomia_del_Prompt.ipynb` — Estructura y buenas prácticas
3. `03_Tecnicas_Prompting.ipynb` — Zero/one/few-shot, CoT, role, context
4. `04_Optimizacion_Prompts.ipynb` — Precisión y anti-alucinaciones
5. `05_Aplicaciones_Practicas.ipynb` — Código, contenido, resumen, traducción
6. `06_Comparacion_Estrategias.ipynb` — Comparación lado a lado
7. `07_Laboratorio_Final.ipynb` — Caso empresarial integrador

## Prerrequisitos

Completar la **Clase 2: La Revolución de los Transformers** antes de iniciar.

## Actividad final — PromptEngineering Lab

```bash
cd ../PromptEngineering_Lab
pip install -r requirements.txt
jupyter notebook notebooks/actividad.ipynb
```

Diseña, prueba y compara estrategias de prompting. Ver rúbrica en [PromptEngineering_Lab/README.md](../PromptEngineering_Lab/README.md).

## Regenerar notebooks

Si modificas `generate_notebooks.py`:

```bash
python generate_notebooks.py
```
