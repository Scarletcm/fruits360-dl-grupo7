"""Entrena UN modelo y guarda historial, tiempo, parámetros y curvas.

Uso (desde la raíz del repositorio):
    python -m src.train --model mobilenetv2
Modelos: cnn, mobilenetv2, efficientnetb0, convnexttiny
"""
import argparse
import json
import os
import random
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src import config as cfg
from src.data import cargar_datasets, listar_particion
from src.models import compilar, construir_modelo, contar_parametros


def fijar_semilla():
    os.environ["PYTHONHASHSEED"] = str(cfg.SEMILLA)
    random.seed(cfg.SEMILLA)
    np.random.seed(cfg.SEMILLA)
    tf.keras.utils.set_random_seed(cfg.SEMILLA)


def graficar_curvas(hist, mejor_epoca, nombre, destino):
    epocas = range(1, len(hist["loss"]) + 1)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, clave, titulo in [(ejes[0], "loss", "Pérdida (loss)"),
                              (ejes[1], "accuracy", "Exactitud (accuracy)")]:
        ax.plot(epocas, hist[clave], marker="o", label="Entrenamiento")
        ax.plot(epocas, hist[f"val_{clave}"], marker="s", label="Validación")
        ax.axvline(mejor_epoca, color="gray", linestyle="--", label=f"Mejor época ({mejor_epoca})")
        ax.set_title(f"{nombre}: {titulo}")
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(cfg.MODELOS))
    ap.add_argument("--epochs", type=int, default=cfg.EPOCAS)
    args = ap.parse_args()

    clave = args.model
    nombre = cfg.MODELOS[clave]["nombre"]
    salida = cfg.RESULTADOS_DIR / clave
    salida.mkdir(parents=True, exist_ok=True)
    cfg.MODELOS_DIR.mkdir(parents=True, exist_ok=True)

    fijar_semilla()
    df = listar_particion()
    datos = cargar_datasets(clave, df)
    modelo = compilar(construir_modelo(clave))
    parametros = contar_parametros(modelo)

    gpus = tf.config.list_physical_devices("GPU")
    gpu_nombres = []
    for g in gpus:
        try:
            gpu_nombres.append(tf.config.experimental.get_device_details(g).get("device_name", g.name))
        except Exception:
            gpu_nombres.append(g.name)
    print(f"\n=== Entrenando {nombre} | GPU: {gpu_nombres or 'NINGUNA (CPU)'} ===")
    print("Parámetros:", parametros)

    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg.PACIENCIA, restore_best_weights=True)]

    inicio = time.perf_counter()
    historial = modelo.fit(datos["Train"], validation_data=datos["Validation"],
                           epochs=args.epochs, callbacks=callbacks, verbose=2)
    tiempo = time.perf_counter() - inicio

    hist = {k: [float(v) for v in vals] for k, vals in historial.history.items()}
    mejor_epoca = int(np.argmin(hist["val_loss"])) + 1

    info = {
        "modelo": nombre,
        "clave": clave,
        "tiempo_entrenamiento_s": round(tiempo, 1),
        "epocas_ejecutadas": len(hist["loss"]),
        "mejor_epoca": mejor_epoca,
        "val_accuracy_mejor_epoca": hist["val_accuracy"][mejor_epoca - 1],
        "val_loss_mejor_epoca": hist["val_loss"][mejor_epoca - 1],
        "parametros": parametros,
        "hiperparametros": {
            "epocas_max": args.epochs, "batch_size": cfg.BATCH_SIZE,
            "optimizador": cfg.OPTIMIZADOR, "learning_rate": cfg.LEARNING_RATE,
            "perdida": cfg.FUNCION_PERDIDA, "paciencia": cfg.PACIENCIA,
            "semilla": cfg.SEMILLA, "margen_train": cfg.USAR_MARGEN_TRAIN,
        },
        "entorno": {"gpu": gpu_nombres, "tensorflow": tf.__version__,
                    "fecha": datetime.now().isoformat(timespec="seconds")},
    }
    (salida / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    (salida / "train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_curvas(hist, mejor_epoca, nombre, salida / "curvas.png")
    modelo.save(cfg.MODELOS_DIR / f"{clave}.keras")

    print(f"\nListo: {nombre} | {len(hist['loss'])} épocas | mejor época {mejor_epoca} | {tiempo:.0f} s")
    print(f"Resultados en: {salida}")


if __name__ == "__main__":
    main()
