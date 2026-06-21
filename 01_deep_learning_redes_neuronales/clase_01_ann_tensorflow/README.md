# Clase 1: Fundamentos de Redes Neuronales Artificiales (ANN) con TensorFlow

## Caso de Estudio
Clasificacion de flores con Iris Dataset.

---

## Informacion General

- Modulo: Cientifico de Datos e Inteligencia Artificial Aplicada
- Unidad: 1
- Clase: 1
- Tema: Fundamentos de Redes Neuronales Artificiales (ANN)
- Tecnologias: Python, TensorFlow, Scikit-Learn, Jupyter Notebook
- Dataset: Iris Dataset

---

## Objetivos de Aprendizaje

Al finalizar esta clase, el estudiante sera capaz de:

- Comprender que es una Red Neuronal Artificial.
- Identificar la estructura de una neurona artificial.
- Comprender el funcionamiento de las capas densas.
- Aplicar funciones de activacion ReLU y Softmax.
- Crear y manipular tensores en TensorFlow.
- Diferenciar tensores constantes y variables.
- Construir una red neuronal basica para clasificacion.
- Entender como una ANN aprende con un dataset real.

---

## 1. Por Que Necesitamos Redes Neuronales

Las redes neuronales se usan cuando una regla fija no alcanza para resolver un problema.

### Ejemplo rapido: deteccion de spam
Entradas posibles:

- Cantidad de enlaces
- Cantidad de imagenes
- Palabras sospechosas
- Longitud del mensaje

Salida:

- Spam
- No Spam

Con reglas simples como `if enlaces > 3` no es suficiente, porque hay miles de combinaciones posibles.

### Idea clave
Necesitamos sistemas capaces de:

- Aprender patrones
- Encontrar relaciones ocultas
- Generalizar en casos nuevos
- Mejorar con la experiencia

---

## 2. Que Es una Red Neuronal Artificial

Una ANN es un modelo matematico inspirado en el cerebro humano. Aprende relaciones entre entradas y salidas a partir de ejemplos.

### Diferencia con programacion tradicional

Programacion tradicional:

- Datos + Reglas -> Resultado

Machine Learning:

- Datos + Resultados -> Aprendizaje de reglas

En ML, el sistema descubre automaticamente las reglas.

---

## 3. Nuestro Caso de Estudio: Iris Dataset

Iris es un dataset clasico de clasificacion multiclase.

### Caracteristicas de entrada

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

### Clases objetivo

- Setosa
- Versicolor
- Virginica

### Objetivo del modelo
Predecir la especie de una flor a partir de sus 4 medidas.

---

## 4. Arquitectura de una Neurona Artificial

Una neurona recibe entradas, aplica pesos, suma un sesgo y pasa el resultado por una funcion de activacion.

Formula base:

$$
z = w_1x_1 + w_2x_2 + w_3x_3 + w_4x_4 + b
$$

Luego:

$$
a = f(z)
$$

Donde:

- $x_i$: entradas
- $w_i$: pesos
- $b$: bias (sesgo)
- $f$: funcion de activacion

---

## 5. Capas de una Red Neuronal

### Capa de entrada
Recibe las 4 variables del Iris.

### Capa oculta
Aprende representaciones internas (patrones).

### Capa de salida
Entrega probabilidades para 3 clases (Setosa, Versicolor, Virginica).

Arquitectura base usada en la clase:

- Entrada (4)
- Dense(8) + ReLU
- Dense(3) + Softmax

---

## 6. Capas Densas (Dense)

Una capa densa conecta cada entrada con cada neurona de la capa siguiente.

Ejemplo:

- Dense(8): transforma 4 entradas en 8 neuronas internas

Esta capa aprende combinaciones utiles de las variables originales.

---

## 7. Funciones de Activacion

### ReLU

$$
\text{ReLU}(x) = \max(0, x)
$$

Ventaja: ayuda a entrenar redes profundas de forma eficiente.

### Softmax
Convierte puntajes finales en probabilidades por clase.

Propiedad clave:

- La suma de probabilidades de todas las clases es 1.

---

## 8. Introduccion a TensorFlow y Tensores

TensorFlow trabaja con tensores.

### Tipos comunes

- Escalar (0D)
- Vector (1D)
- Matriz (2D)
- Tensor nD

### Tensores constantes
No cambian durante la ejecucion:

- `tf.constant(...)`

### Tensores variables
Pueden modificarse y se usan en entrenamiento:

- `tf.Variable(...)`

---

## 9. Flujo de la Practica de Clase

1. Cargar Iris Dataset.
2. Separar entrenamiento y prueba.
3. Estandarizar variables.
4. Construir ANN simple con TensorFlow.
5. Entrenar el modelo.
6. Evaluar accuracy en test.
7. Hacer predicciones de ejemplo.

---

## 10. Resultado Esperado

Al finalizar, el estudiante deberia poder explicar y ejecutar todo el flujo basico de clasificacion con una ANN:

- Entender la teoria minima
- Implementar el modelo en TensorFlow
- Interpretar metricas de desempeno

---

## Recurso Practico de la Clase

Notebook de la sesion:

- U1_C1_fundamentos_ann_tensorflow_v1.ipynb
