"""Las cuatro arquitecturas del proyecto (sección 9 del informe)."""
import tensorflow as tf
from tensorflow.keras import layers, models

from src import config as cfg


def crear_augmentation():
    """Idéntico a la sección 7.4. Solo actúa durante el entrenamiento."""
    return tf.keras.Sequential([
        layers.RandomRotation(0.04, fill_mode="constant", fill_value=255.0),
        layers.RandomTranslation(0.03, 0.03, fill_mode="constant", fill_value=255.0),
        layers.RandomZoom(height_factor=(-0.04, 0.0), width_factor=(-0.04, 0.0),
                          fill_mode="constant", fill_value=255.0),
    ], name="data_augmentation")


def construir_cnn():
    entrada = layers.Input((100, 100, 3))
    x = crear_augmentation()(entrada)
    x = layers.Rescaling(1.0 / 255)(x)
    for filtros in (32, 64, 128):
        x = layers.Conv2D(filtros, 3, activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    salida = layers.Dense(cfg.NUM_CLASES, activation="softmax")(x)
    return models.Model(entrada, salida, name="CNN_propia")


def _construir_transfer(nombre, constructor, preprocesar=None):
    """Transfer learning por extracción de características (backbone congelado)."""
    entrada = layers.Input((224, 224, 3))
    x = crear_augmentation()(entrada)
    if preprocesar is not None:
        x = preprocesar(x)
    base = constructor(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Dense(cfg.NUM_CLASES, activation="softmax")(x)
    return models.Model(entrada, salida, name=nombre)


def construir_mobilenetv2():
    return _construir_transfer("MobileNetV2", tf.keras.applications.MobileNetV2,
                               layers.Rescaling(1.0 / 127.5, offset=-1))


def construir_efficientnetb0():
    return _construir_transfer("EfficientNetB0", tf.keras.applications.EfficientNetB0)


def construir_convnexttiny():
    return _construir_transfer("ConvNeXtTiny", tf.keras.applications.ConvNeXtTiny)


CONSTRUCTORES = {
    "cnn": construir_cnn,
    "mobilenetv2": construir_mobilenetv2,
    "efficientnetb0": construir_efficientnetb0,
    "convnexttiny": construir_convnexttiny,
}


def construir_modelo(clave):
    if clave not in CONSTRUCTORES:
        raise ValueError(f"Modelo desconocido: {clave}. Opciones: {list(CONSTRUCTORES)}")
    tf.keras.utils.set_random_seed(cfg.SEMILLA)     # misma inicialización siempre
    return CONSTRUCTORES[clave]()


def compilar(modelo):
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE),
        loss=cfg.FUNCION_PERDIDA,
        metrics=["accuracy"],
    )
    return modelo


def contar_parametros(modelo):
    entrenables = sum(int(tf.size(w)) for w in modelo.trainable_weights)
    total = int(modelo.count_params())
    return {"totales": total, "entrenables": entrenables, "no_entrenables": total - entrenables}
