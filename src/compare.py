"""Une los resultados de los 4 modelos en la tabla comparativa obligatoria.

Uso:
    python -m src.compare
Lee results/<modelo>/train_info.json y metrics.json; escribe
results/tabla_comparativa.csv y results/tabla_comparativa.md
"""
import json

import pandas as pd

from src import config as cfg


def leer_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def main():
    filas = []
    for clave, datos in cfg.MODELOS.items():
        carpeta = cfg.RESULTADOS_DIR / clave
        info = leer_json(carpeta / "train_info.json")
        met = leer_json(carpeta / "metrics.json")
        if info is None:
            print(f"[aviso] Falta train_info.json de {datos['nombre']}")
            continue
        fila = {
            "Modelo": datos["nombre"],
            "Accuracy": met["accuracy"] if met else None,
            "Precision": met["precision_macro"] if met else None,
            "Recall": met["recall_macro"] if met else None,
            "F1-score": met["f1_macro"] if met else None,
            "ROC-AUC": met["roc_auc_ovr_macro"] if met else None,
            "Tiempo de entrenamiento (s)": info["tiempo_entrenamiento_s"],
            "Parámetros totales": info["parametros"]["totales"],
            "Parámetros entrenables": info["parametros"]["entrenables"],
            "Épocas": info["epocas_ejecutadas"],
            "Mejor época": info["mejor_epoca"],
            "Val accuracy": info["val_accuracy_mejor_epoca"],
        }
        # Generalización: diferencia entre validación y prueba
        fila["Brecha Val-Test"] = (fila["Val accuracy"] - fila["Accuracy"]) if met else None
        filas.append(fila)

    if not filas:
        raise SystemExit("No hay resultados todavía. Entrena al menos un modelo.")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(cfg.RESULTADOS_DIR / "tabla_comparativa.csv", index=False, encoding="utf-8-sig")

    texto = tabla.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Val accuracy", "Brecha Val-Test"]:
        texto[col] = texto[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    for col in ["Parámetros totales", "Parámetros entrenables"]:
        texto[col] = texto[col].map(lambda v: f"{v:,}".replace(",", "."))
    columnas = list(texto.columns)
    md = ["| " + " | ".join(columnas) + " |", "|" + "---|" * len(columnas)]
    md += ["| " + " | ".join(str(v) for v in fila) + " |" for fila in texto.itertuples(index=False)]
    (cfg.RESULTADOS_DIR / "tabla_comparativa.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\nGuardado en {cfg.RESULTADOS_DIR}")


if __name__ == "__main__":
    main()
