"""Carga de datos: partición fija desde carpetas y pipelines tf.data."""
import tensorflow as tf

from src import config as cfg
from src.particion import listar_particion  # noqa: F401  (se re-exporta)

AUTOTUNE = tf.data.AUTOTUNE


def _leer_imagen(ruta, etiqueta, tamano):
    contenido = tf.io.read_file(ruta)
    imagen = tf.io.decode_image(contenido, channels=3, expand_animations=False)
    imagen.set_shape([None, None, 3])
    imagen = tf.image.resize(imagen, tamano, method="bilinear")
    return tf.cast(imagen, tf.float32), etiqueta


def _con_margen(imagen, etiqueta, tamano, escala=cfg.ESCALA_MARGEN):
    alto, ancho = tamano
    nuevo_alto, nuevo_ancho = int(alto * escala), int(ancho * escala)
    reducida = tf.image.resize(imagen, (nuevo_alto, nuevo_ancho), method="bilinear")
    arriba = (alto - nuevo_alto) // 2
    izquierda = (ancho - nuevo_ancho) // 2
    relleno = [[arriba, alto - nuevo_alto - arriba],
               [izquierda, ancho - nuevo_ancho - izquierda],
               [0, 0]]
    return tf.pad(reducida, relleno, constant_values=255.0), etiqueta


def crear_dataset(df, conjunto, tamano, entrenar=False):
    """Dataset en escala 0-255; cada modelo normaliza dentro de la red."""
    sub = df[df["conjunto"] == conjunto]
    ds = tf.data.Dataset.from_tensor_slices(
        (sub["ruta"].tolist(), sub["etiqueta"].astype("int32").tolist()))
    if entrenar:                                    # solo Train se mezcla
        ds = ds.shuffle(len(sub), seed=cfg.SEMILLA, reshuffle_each_iteration=True)
    ds = ds.map(lambda r, e: _leer_imagen(r, e, tamano), num_parallel_calls=AUTOTUNE)
    if entrenar and cfg.USAR_MARGEN_TRAIN:
        ds = ds.map(lambda i, e: _con_margen(i, e, tamano), num_parallel_calls=AUTOTUNE)
    return ds.batch(cfg.BATCH_SIZE).prefetch(AUTOTUNE)


def cargar_datasets(clave_modelo, df=None):
    """Devuelve {'Train': ds, 'Validation': ds, 'Test': ds} para el modelo indicado."""
    if df is None:
        df = listar_particion()
    tamano = cfg.MODELOS[clave_modelo]["tamano"]
    return {c: crear_dataset(df, c, tamano, entrenar=(c == "Train")) for c in cfg.CONJUNTOS}
