"""Lectura y validación de la partición fija (sin depender de TensorFlow)."""
from pathlib import Path

import pandas as pd

from src import config as cfg


def listar_particion(data_dir=cfg.DATA_DIR):
    """Devuelve un DataFrame (conjunto, clase, etiqueta, ruta) y valida cantidades."""
    registros = []
    for conjunto in cfg.CONJUNTOS:
        for clase in cfg.CLASES:
            carpeta = Path(data_dir) / conjunto / clase
            if not carpeta.is_dir():
                raise FileNotFoundError(
                    f"No existe {carpeta}. Ejecuta scripts/prepare_data.py (ver data/README.md)."
                )
            for archivo in sorted(carpeta.iterdir()):
                if archivo.suffix.lower() in cfg.EXTENSIONES:
                    registros.append({
                        "conjunto": conjunto,
                        "clase": clase,
                        "etiqueta": cfg.CLASES.index(clase),
                        "ruta": str(archivo),
                    })
    df = pd.DataFrame(registros)
    for conjunto, esperado in cfg.CANTIDADES_ESPERADAS.items():
        real = int((df["conjunto"] == conjunto).sum())
        if real != esperado:
            raise ValueError(f"{conjunto}: {real} imágenes (esperadas {esperado}).")
    return df
