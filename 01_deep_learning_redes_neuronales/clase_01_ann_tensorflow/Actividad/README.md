# Actividad: ANN con Dataset de Kaggle

## Contexto
En esta actividad debes aplicar el flujo visto en el ejemplo de clase para construir un modelo de Red Neuronal Artificial (ANN) con un dataset real tomado de Kaggle.

Base de referencia:

- Revisa el notebook del ejemplo en la carpeta `Ejemplo` y replica su estructura general (carga, limpieza, preprocesamiento, entrenamiento, evaluacion y graficas).

---

## Objetivo
Seleccionar un dataset de Kaggle y entrenar una ANN para resolver un problema de clasificacion o regresion, justificando las decisiones tecnicas del pipeline.

---

## Requisitos de la actividad
1. Selecciona un dataset de Kaggle y documenta su enlace.
2. Define claramente:
   - Variable objetivo (target).
   - Tipo de problema (clasificacion o regresion).
3. Realiza analisis exploratorio minimo.
4. Preprocesa los datos correctamente.
5. Construye y entrena una ANN en TensorFlow/Keras.
6. Evalua el modelo con metricas adecuadas.
7. Incluye graficas de entrenamiento (loss/accuracy o la equivalente para regresion).
8. Presenta conclusiones tecnicas y mejoras futuras.

---

## Que se busca en el dataset (ejemplos de criterios)

### 1) Objetivo claro
- Bueno: `churn` (si/no), `loan_status` (approved/rejected), `price` (valor numerico).
- Evitar: datasets sin columna objetivo clara o con variables ambiguas.

### 2) Tamano util para entrenar
- Recomendado: al menos cientos o miles de filas.
- Evitar: datasets demasiado pequenos para entrenar una ANN de forma estable.

### 3) Calidad de datos
- Revisar nulos, duplicados, valores extremos y formato de columnas.
- Ejemplo: columnas categoricas con espacios extra o etiquetas inconsistentes.

### 4) Mezcla de variables
- Ideal: dataset con variables numericas y categoricas para practicar preprocesamiento real.
- Ejemplo: edad, ingresos, puntaje crediticio + genero, ocupacion, nivel educativo.

### 5) Riesgo de leakage
- Evitar columnas que revelen directamente la respuesta.
- Ejemplo: usar una columna posterior al evento que contiene la decision final.

### 6) Balance de clases (si es clasificacion)
- Revisar distribucion del target.
- Si hay fuerte desbalance, justificar estrategia (class_weight, umbral, metricas por clase).

---

## Estructura sugerida del notebook
1. Portada y contexto del dataset.
2. Importaciones.
3. Carga y limpieza de datos.
4. EDA rapido.
5. Preprocesamiento.
6. Modelo ANN.
7. Entrenamiento.
8. Evaluacion y graficas.
9. Conclusiones.

---
