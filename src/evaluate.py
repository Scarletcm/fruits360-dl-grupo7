"""Evalúa UN modelo ya entrenado sobre el conjunto Test (una sola vez).

Uso:
    python -m src.evaluate --model mobilenetv2
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from src import config as cfg
from src.data import cargar_datasets, listar_particion


def graficar_matriz(cm, nombre, destino):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(cfg.NUM_CLASES), cfg.CLASES, rotation=30, ha="right")
    ax.set_yticks(range(cfg.NUM_CLASES), cfg.CLASES)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusión: {nombre}")
    umbral = cm.max() / 2
    for i in range(cfg.NUM_CLASES):
        for j in range(cfg.NUM_CLASES):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > umbral else "black")
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(cfg.MODELOS))
    args = ap.parse_args()

    clave = args.model
    nombre = cfg.MODELOS[clave]["nombre"]
    salida = cfg.RESULTADOS_DIR / clave
    salida.mkdir(parents=True, exist_ok=True)

    modelo = tf.keras.models.load_model(cfg.MODELOS_DIR / f"{clave}.keras", compile=False)
    df = listar_particion()
    ds_test = cargar_datasets(clave, df)["Test"]

    probs = modelo.predict(ds_test, verbose=0)
    y_real = np.concatenate([y.numpy() for _, y in ds_test])
    y_pred = probs.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_real, y_pred, average="macro", zero_division=0)
    metricas = {
        "modelo": nombre,
        "accuracy": float(accuracy_score(y_real, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc_ovr_macro": float(roc_auc_score(y_real, probs, multi_class="ovr", average="macro")),
        "imagenes_test": int(len(y_real)),
    }
    (salida / "metrics.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    cm = confusion_matrix(y_real, y_pred, labels=range(cfg.NUM_CLASES))
    pd.DataFrame(cm, index=cfg.CLASES, columns=cfg.CLASES).to_csv(
        salida / "matriz_confusion.csv", encoding="utf-8-sig")
    graficar_matriz(cm, nombre, salida / "matriz_confusion.png")

    reporte = classification_report(y_real, y_pred, target_names=cfg.CLASES, digits=4, zero_division=0)
    (salida / "reporte_clasificacion.txt").write_text(reporte, encoding="utf-8")

    # Errores para el análisis de la sección 14
    rutas = df[df["conjunto"] == "Test"]["ruta"].tolist()
    errores = [{"ruta": rutas[i], "real": cfg.CLASES[y_real[i]], "predicha": cfg.CLASES[y_pred[i]],
                "confianza": float(probs[i].max())}
               for i in np.flatnonzero(y_real != y_pred)]
    pd.DataFrame(errores, columns=["ruta", "real", "predicha", "confianza"]).to_csv(
        salida / "errores_test.csv", index=False, encoding="utf-8-sig")

    print(f"\n=== {nombre} (Test, {len(y_real)} imágenes) ===")
    print(reporte)
    print(json.dumps(metricas, indent=2, ensure_ascii=False))
    print(f"Errores: {len(errores)}")


if __name__ == "__main__":
    main()
