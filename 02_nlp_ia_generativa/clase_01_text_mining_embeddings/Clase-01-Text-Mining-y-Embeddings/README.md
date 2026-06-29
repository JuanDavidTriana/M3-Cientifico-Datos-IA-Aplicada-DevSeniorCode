# Clase 01: Text Mining y Embeddings

Repositorio de apoyo para una clase de aproximadamente 4 horas del módulo **Científico de Datos e Inteligencia Artificial Aplicada**.

## Estructura del proyecto

```text
Clase-01-Text-Mining-y-Embeddings/
│
├── README.md
├── requirements.txt
├── datasets/
│   ├── frases.csv
│   ├── opiniones_clientes.csv
│   ├── noticias.csv
│   └── peliculas.csv
└── notebooks/
    ├── 01_Introduccion_Texto.ipynb
    ├── 02_Preprocesamiento.ipynb
    ├── 03_Tokenizacion.ipynb
    ├── 04_Padding.ipynb
    ├── 05_Embedding_Layer.ipynb
    ├── 06_Mini_Clasificador_Sentimientos.ipynb
    └── 07_Laboratorio_Final.ipynb
```

## Instalación de dependencias

1. Crear y activar un entorno virtual (opcional, recomendado).
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Cómo ejecutar los notebooks

Desde la raíz del proyecto:

```bash
jupyter notebook
```

Abrir la carpeta `notebooks/` y ejecutar cada notebook en orden.

## Orden recomendado de estudio

1. `01_Introduccion_Texto.ipynb`
2. `02_Preprocesamiento.ipynb`
3. `03_Tokenizacion.ipynb`
4. `04_Padding.ipynb`
5. `05_Embedding_Layer.ipynb`
6. `06_Mini_Clasificador_Sentimientos.ipynb`
7. `07_Laboratorio_Final.ipynb`

## Objetivos de aprendizaje por notebook

- **01 Introducción al Texto**: comprender por qué el texto debe transformarse antes de modelar.
- **02 Preprocesamiento**: aplicar limpieza paso a paso con expresiones regulares y pandas.
- **03 Tokenización**: convertir texto a índices numéricos con `Tokenizer`.
- **04 Padding**: estandarizar longitudes de secuencias para redes neuronales.
- **05 Embedding Layer**: construir una red simple con `Embedding`, entender shapes y parámetros.
- **06 Mini Clasificador de Sentimientos**: entrenar un modelo completo de extremo a extremo.
- **07 Laboratorio Final**: integrar todo el pipeline y experimentar con hiperparámetros.

## Notas didácticas

- Cada notebook es independiente: importa sus propias librerías y carga datos por rutas relativas.
- Se muestran resultados intermedios con `print()` o `display()`.
- Todos los notebooks incluyen preguntas de reflexión para discusión en clase.
- Los datasets son pequeños y realistas para favorecer la comprensión conceptual.
