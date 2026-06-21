# Ejemplo ANN: Aprobacion de Prestamos

## Contexto del Dataset
Este ejemplo utiliza el dataset **Loan Approval Prediction Dataset** (Kaggle):

- Fuente: https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset
- Objetivo de negocio: predecir si una solicitud de prestamo sera **Approved** o **Rejected**.
- Tipo de problema: clasificacion binaria.

### Variables relevantes
- Variables numericas: ingresos, monto del prestamo, plazo, cibil score y activos.
- Variables categoricas: nivel educativo y tipo de empleo (self employed).
- Variable objetivo: `loan_status`.

## Objetivo del Ejemplo
Construir una red neuronal artificial (ANN) en TensorFlow/Keras para clasificar solicitudes de prestamo.

El flujo incluye:
1. Carga y limpieza de datos.
2. Preprocesamiento (One-Hot Encoding + escalado).
3. Entrenamiento de una ANN.
4. Evaluacion con accuracy, reporte de clasificacion y matriz de confusion.
5. Visualizacion de curvas de entrenamiento.

## Enunciado del Reto (Actividad para Estudiantes)
Actuas como cientifico/a de datos en una entidad financiera. Debes construir un modelo ANN que ayude a predecir la aprobacion de creditos para asistir al equipo de analisis.

### Requerimientos del reto
1. Entrenar un modelo ANN para `loan_status`.
2. Alcanzar al menos **85% de accuracy en test** (meta sugerida).
3. Mostrar y explicar:
   - Curvas de entrenamiento (`accuracy` y `loss`).
   - Matriz de confusion.
   - Precision, recall y F1-score.
4. Probar al menos **2 configuraciones** distintas de arquitectura (por ejemplo, capas/neuronas/regularizacion).
5. Redactar una conclusion tecnica:
   - Que modelo funciono mejor.
   - Si hubo overfitting.
   - Que mejoras implementarias en una siguiente iteracion.

## Archivo principal de ejemplo
- `loan_ann_example.ipynb`

