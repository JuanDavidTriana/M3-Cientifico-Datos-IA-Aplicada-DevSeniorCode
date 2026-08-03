"""
Cliente de prueba simple para el servicio de inferencia.

Genera una imagen sintética (ruido) de 28x28, la envía al endpoint /predict
y muestra la respuesta. Útil para verificar rápidamente que el contenedor
está funcionando, sin necesidad de un dataset real.

Uso (con la API corriendo en localhost:8000):
    python test_api.py
    python test_api.py --host http://localhost:8000
"""
import argparse

import numpy as np
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    args = parser.parse_args()

    print(f"Probando servicio en {args.host}")

    salud = requests.get(f"{args.host}/health", timeout=10)
    print("GET /health ->", salud.status_code, salud.json())

    imagen_aleatoria = (np.random.rand(28, 28) * 255).tolist()
    respuesta = requests.post(
        f"{args.host}/predict",
        json={"imagen": imagen_aleatoria},
        timeout=10,
    )
    print("POST /predict ->", respuesta.status_code, respuesta.json())


if __name__ == "__main__":
    main()
