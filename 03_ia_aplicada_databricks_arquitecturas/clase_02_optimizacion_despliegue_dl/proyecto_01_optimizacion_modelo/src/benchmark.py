"""
Compara el modelo base (.h5) contra el modelo optimizado (.tflite):
tamaño en disco, latencia de inferencia y accuracy en el set de prueba.

Uso:
    python src/benchmark.py
"""
import os

# Debe fijarse ANTES de importar tensorflow, para que keras.models.load_model
# pueda leer los .h5 guardados por train_model.py / optimize_model.py.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
BASE_MODEL_PATH = os.path.join(MODELS_DIR, "modelo_base.h5")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "modelo_optimizado.tflite")
REPORT_PATH = os.path.join(MODELS_DIR, "comparacion.md")

N_MUESTRAS_LATENCIA = 200


def cargar_datos_test():
    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    return x_test, y_test


def tamano_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def evaluar_keras(path, x_test, y_test):
    modelo = keras.models.load_model(path)
    _, acc = modelo.evaluate(x_test, y_test, verbose=0)

    # Latencia: inferencia de a una muestra, N veces
    muestras = x_test[:N_MUESTRAS_LATENCIA]
    inicio = time.perf_counter()
    for i in range(len(muestras)):
        modelo.predict(muestras[i:i + 1], verbose=0)
    duracion = time.perf_counter() - inicio
    latencia_ms = (duracion / len(muestras)) * 1000

    return acc, latencia_ms


def evaluar_tflite(path, x_test, y_test):
    interprete = tf.lite.Interpreter(model_path=path)
    interprete.allocate_tensors()
    entrada = interprete.get_input_details()[0]
    salida = interprete.get_output_details()[0]

    # Accuracy sobre todo el set de test
    aciertos = 0
    for i in range(len(x_test)):
        muestra = x_test[i:i + 1].astype(entrada["dtype"])
        interprete.set_tensor(entrada["index"], muestra)
        interprete.invoke()
        pred = np.argmax(interprete.get_tensor(salida["index"])[0])
        if pred == y_test[i]:
            aciertos += 1
    acc = aciertos / len(x_test)

    # Latencia sobre N_MUESTRAS_LATENCIA
    muestras = x_test[:N_MUESTRAS_LATENCIA].astype(entrada["dtype"])
    inicio = time.perf_counter()
    for i in range(len(muestras)):
        interprete.set_tensor(entrada["index"], muestras[i:i + 1])
        interprete.invoke()
        _ = interprete.get_tensor(salida["index"])
    duracion = time.perf_counter() - inicio
    latencia_ms = (duracion / len(muestras)) * 1000

    return acc, latencia_ms


def main():
    for path in (BASE_MODEL_PATH, TFLITE_MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No se encontró {path}. Ejecuta antes train_model.py y optimize_model.py."
            )

    x_test, y_test = cargar_datos_test()

    print("Evaluando modelo base (.h5)...")
    acc_base, lat_base = evaluar_keras(BASE_MODEL_PATH, x_test, y_test)
    size_base = tamano_mb(BASE_MODEL_PATH)

    print("Evaluando modelo optimizado (.tflite)...")
    acc_opt, lat_opt = evaluar_tflite(TFLITE_MODEL_PATH, x_test, y_test)
    size_opt = tamano_mb(TFLITE_MODEL_PATH)

    filas = [
        ("Tamaño en disco (MB)", f"{size_base:.2f}", f"{size_opt:.2f}", f"{(1 - size_opt / size_base) * 100:.1f}% más chico"),
        ("Accuracy en test", f"{acc_base:.4f}", f"{acc_opt:.4f}", f"{(acc_base - acc_opt) * 100:+.2f} pts"),
        ("Latencia promedio (ms/inferencia)", f"{lat_base:.3f}", f"{lat_opt:.3f}", f"{(1 - lat_opt / lat_base) * 100:.1f}% más rápido"),
    ]

    print("\n=== Comparación modelo base vs modelo optimizado ===")
    header = f"{'Métrica':35} {'Base (.h5)':>15} {'Optimizado (.tflite)':>22} {'Diferencia':>20}"
    print(header)
    print("-" * len(header))
    for nombre, base, opt, diff in filas:
        print(f"{nombre:35} {base:>15} {opt:>22} {diff:>20}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Comparación: modelo base vs modelo optimizado\n\n")
        f.write("| Métrica | Base (.h5) | Optimizado (.tflite) | Diferencia |\n")
        f.write("|---|---|---|---|\n")
        for nombre, base, opt, diff in filas:
            f.write(f"| {nombre} | {base} | {opt} | {diff} |\n")

    print(f"\nReporte guardado en: {REPORT_PATH}")


if __name__ == "__main__":
    main()
