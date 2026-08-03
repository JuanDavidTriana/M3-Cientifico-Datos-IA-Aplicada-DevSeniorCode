"""
Compara en vivo los dos servicios levantados con docker-compose:

    modelo-base        -> http://localhost:8001
    modelo-optimizado   -> http://localhost:8002

Mide, para cada servicio: latencia promedio por petición y throughput
(peticiones/segundo) enviando el mismo lote de imágenes sintéticas a ambos.
También reporta el tamaño en disco de cada artefacto de modelo, si están
disponibles localmente.

Uso (con 'docker compose up --build' corriendo en otra terminal):
    python benchmark/compare.py
    python benchmark/compare.py --n 100
"""
import argparse
import os
import time

import numpy as np
import requests

BASE_URL = "http://localhost:8001"
OPT_URL = "http://localhost:8002"

MODEL_BASE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "servicio_baseline", "model", "modelo_base.h5"
)
MODEL_OPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "servicio_optimizado", "model", "modelo_optimizado.tflite"
)

REPORT_PATH = os.path.join(os.path.dirname(__file__), "reporte.md")


def esperar_servicio(url, intentos=20, espera=2):
    for _ in range(intentos):
        try:
            r = requests.get(f"{url}/health", timeout=3)
            if r.status_code == 200 and r.json().get("modelo_cargado"):
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(espera)
    return False


def generar_lote(n):
    return [(np.random.rand(28, 28) * 255).tolist() for _ in range(n)]


def medir_servicio(url, lote):
    latencias = []
    inicio_total = time.perf_counter()
    for imagen in lote:
        inicio = time.perf_counter()
        r = requests.post(f"{url}/predict", json={"imagen": imagen}, timeout=10)
        r.raise_for_status()
        latencias.append((time.perf_counter() - inicio) * 1000)
    duracion_total = time.perf_counter() - inicio_total

    return {
        "latencia_promedio_ms": sum(latencias) / len(latencias),
        "latencia_p95_ms": sorted(latencias)[int(len(latencias) * 0.95) - 1],
        "throughput_req_s": len(lote) / duracion_total,
    }


def tamano_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Número de peticiones por servicio")
    args = parser.parse_args()

    print("Esperando a que ambos servicios estén listos...")
    base_lista = esperar_servicio(BASE_URL)
    opt_lista = esperar_servicio(OPT_URL)

    if not base_lista or not opt_lista:
        print(
            "Alguno de los servicios no respondió a tiempo. Verifica que "
            "'docker compose up --build' esté corriendo y que los modelos "
            "estén copiados en cada carpeta model/."
        )
        return

    print(f"Generando lote de {args.n} imágenes sintéticas...")
    lote = generar_lote(args.n)

    print("Midiendo servicio BASE (puerto 8001)...")
    metricas_base = medir_servicio(BASE_URL, lote)

    print("Midiendo servicio OPTIMIZADO (puerto 8002)...")
    metricas_opt = medir_servicio(OPT_URL, lote)

    size_base = tamano_mb(MODEL_BASE_PATH)
    size_opt = tamano_mb(MODEL_OPT_PATH)

    print("\n=== Resultado del benchmark ===")
    print(f"{'Métrica':30} {'Base':>15} {'Optimizado':>15}")
    print("-" * 62)
    print(f"{'Latencia promedio (ms)':30} {metricas_base['latencia_promedio_ms']:>15.2f} {metricas_opt['latencia_promedio_ms']:>15.2f}")
    print(f"{'Latencia p95 (ms)':30} {metricas_base['latencia_p95_ms']:>15.2f} {metricas_opt['latencia_p95_ms']:>15.2f}")
    print(f"{'Throughput (req/s)':30} {metricas_base['throughput_req_s']:>15.2f} {metricas_opt['throughput_req_s']:>15.2f}")
    if size_base and size_opt:
        print(f"{'Tamaño del modelo (MB)':30} {size_base:>15.2f} {size_opt:>15.2f}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Reporte de benchmark — modelo base vs modelo optimizado\n\n")
        f.write(f"Peticiones por servicio: {args.n}\n\n")
        f.write("| Métrica | Base | Optimizado |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Latencia promedio (ms) | {metricas_base['latencia_promedio_ms']:.2f} | {metricas_opt['latencia_promedio_ms']:.2f} |\n")
        f.write(f"| Latencia p95 (ms) | {metricas_base['latencia_p95_ms']:.2f} | {metricas_opt['latencia_p95_ms']:.2f} |\n")
        f.write(f"| Throughput (req/s) | {metricas_base['throughput_req_s']:.2f} | {metricas_opt['throughput_req_s']:.2f} |\n")
        if size_base and size_opt:
            f.write(f"| Tamaño del modelo (MB) | {size_base:.2f} | {size_opt:.2f} |\n")

    print(f"\nReporte guardado en: {REPORT_PATH}")


if __name__ == "__main__":
    main()
