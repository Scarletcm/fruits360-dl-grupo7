"""Configuración central del proyecto.

Única fuente de verdad: los cuatro modelos usan SIEMPRE estos valores.
Si algo se cambia aquí, se cambia para todos (hacerlo mediante Pull Request).
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FRUTAS_DATA_DIR", RAIZ / "data" / "fruits360_preprocesado"))
RESULTADOS_DIR = RAIZ / "results"
MODELOS_DIR = RAIZ / "models"

# --- Dataset ---
SEMILLA = 42
CLASES = ["Apple Red 1", "Banana 1", "Orange 1", "Strawberry 1"]
NUM_CLASES = len(CLASES)
CONJUNTOS = ["Train", "Validation", "Test"]
EXTENSIONES = (".jpg", ".jpeg", ".png")
CANTIDADES_ESPERADAS = {"Train": 1562, "Validation": 391, "Test": 654}

# --- Hiperparámetros comunes (Tabla 12 del informe) ---
BATCH_SIZE = 32
EPOCAS = 15            # máximo; EarlyStopping puede detener antes
LEARNING_RATE = 1e-3
PACIENCIA = 3          # EarlyStopping sobre val_loss
OPTIMIZADOR = "Adam"
FUNCION_PERDIDA = "sparse_categorical_crossentropy"

# Margen blanco del 82 % en Train (sección 7.4 del informe). Decidir en grupo.
USAR_MARGEN_TRAIN = False
ESCALA_MARGEN = 0.82

# --- Modelos ---
MODELOS = {
    "cnn": {"nombre": "CNN propia", "tamano": (100, 100)},
    "mobilenetv2": {"nombre": "MobileNetV2", "tamano": (224, 224)},
    "efficientnetb0": {"nombre": "EfficientNetB0", "tamano": (224, 224)},
    "convnexttiny": {"nombre": "ConvNeXtTiny", "tamano": (224, 224)},
}
